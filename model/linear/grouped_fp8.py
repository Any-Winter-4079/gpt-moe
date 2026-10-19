from __future__ import annotations

from typing import Any, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .deepgemm import grouped_fp8_wgrad


@triton.jit
def _inverse_scale(amax):
    # E4M3 with current, power-of-two scales
    scale = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    return tl.where(amax == 0, 1.0, tl.maximum(scale, 2.0 ** -126))


@triton.jit
def _quantize_activations_kernel(
    X, Row, Column, RowScale, ColumnScale, Offsets, PaddedOffsets,
    M: tl.constexpr, K: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    stride_m: tl.constexpr, stride_k: tl.constexpr, SAVE_COLUMN: tl.constexpr,
    DEEPGEMM: tl.constexpr, COLUMN_M: tl.constexpr, WRITE_OFFSETS: tl.constexpr,
):
    tile = tl.program_id(0)
    col = tl.program_id(1) * 128 + tl.arange(0, 128)
    start = 0
    first_tile = 0
    first_scale_word = 0
    for expert in range(NUM_EXPERTS):
        end = tl.load(Offsets + expert)
        last_tile = first_tile + tl.cdiv(end - start, 128)
        if (tile >= first_tile) & (tile < last_tile):
            row = (tile - first_tile) * 128 + tl.arange(0, 128)
            mask = (start + row[:, None] < end) & (col[None, :] < K)
            values = tl.load(
                X + (start + row[:, None]) * stride_m + col[None, :] * stride_k,
                mask=mask, other=0.0,
            ).to(tl.float32)
            row_scale = _inverse_scale(tl.max(tl.abs(values), axis=1))
            quantized = (values / row_scale[:, None]).to(tl.float8e4nv)
            tl.store(Row + (start + row[:, None]) * K + col[None, :], quantized, mask=mask)
            tl.store(
                RowScale + (start + row) * tl.cdiv(K, 128) + tl.program_id(1),
                row_scale, mask=start + row < end,
            )
            if SAVE_COLUMN:
                # transpose quantization uses 128 tokens from the same expert per scale
                column_scale = _inverse_scale(tl.max(tl.abs(values), axis=0))
                quantized_column = (values / column_scale[None, :]).to(tl.float8e4nv)
                if DEEPGEMM:
                    # pad each expert to 128 tokens while writing the transposed operand
                    tl.store(
                        Column + col[None, :] * COLUMN_M + first_tile * 128 + row[:, None],
                        quantized_column, mask=col[None, :] < K,
                    )
                    # four scale exponents share one int32; each expert starts a new word
                    local_tile = tile - first_tile
                    scale_row = first_scale_word + local_tile // 4
                    scale_byte = local_tile % 4
                    scale_ptr = ColumnScale.to(tl.pointer_type(tl.uint8)) + (scale_row * K + col) * 4
                    exponent = (column_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
                    tl.store(scale_ptr + scale_byte, exponent, mask=col < K)
                    if tile == last_tile - 1:
                        for padding in tl.static_range(1, 4):
                            tl.store(scale_ptr + scale_byte + padding, 0, mask=(col < K) & (scale_byte + padding < 4))
                else:
                    tl.store(
                        Column + col[None, :] * M + start + row[:, None], quantized_column, mask=mask,
                    )
                    scale_row = start // 128 + expert + tile - first_tile
                    tl.store(ColumnScale + scale_row * K + col, column_scale, mask=col < K)
        if WRITE_OFFSETS:
            if (tile == 0) & (tl.program_id(1) == 0):
                tl.store(PaddedOffsets + expert, first_tile * 128 + end - start)
        first_scale_word += tl.cdiv(end - start, 512)
        first_tile = last_tile
        start = end


@triton.jit
def _quantize_weights_kernel(
    W, Quantized, Scale,
    N: tl.constexpr, K: tl.constexpr,
    stride_e: tl.constexpr, stride_n: tl.constexpr, stride_k: tl.constexpr,
):
    expert = tl.program_id(0)
    row = tl.program_id(1) * 128 + tl.arange(0, 128)
    col = tl.program_id(2) * 128 + tl.arange(0, 128)
    mask = (row[:, None] < N) & (col[None, :] < K)
    values = tl.load(
        W + expert * stride_e + row[:, None] * stride_n + col[None, :] * stride_k,
        mask=mask, other=0.0,
    ).to(tl.float32)
    scale = _inverse_scale(tl.max(tl.max(tl.abs(values), axis=1), axis=0))
    quantized = (values / scale).to(tl.float8e4nv)
    tl.store(Quantized + expert * N * K + row[:, None] * K + col[None, :], quantized, mask=mask)
    tl.store(
        Scale + (expert * tl.cdiv(N, 128) + tl.program_id(1)) * tl.cdiv(K, 128) + tl.program_id(2),
        scale,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=3, num_warps=4)
        for bm, bn in [(64, 64), (64, 128), (128, 64)]
    ],
    key=["M", "N", "K", "NUM_EXPERTS", "GROUP_K", "a_stride_m", "a_stride_k", "b_stride_k", "b_stride_n"],
)
@triton.jit
def _grouped_fp8_mm_kernel(
    A, B, AScale, BScale, C, Offsets,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    a_stride_m: tl.constexpr, a_stride_k: tl.constexpr,
    b_stride_e: tl.constexpr, b_stride_k: tl.constexpr, b_stride_n: tl.constexpr,
    as_stride_m: tl.constexpr, as_stride_k: tl.constexpr,
    bs_stride_e: tl.constexpr, bs_stride_k: tl.constexpr, bs_stride_n: tl.constexpr,
    NUM_EXPERTS: tl.constexpr, NUM_PROGRAMS: tl.constexpr, GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    first_tile = 0
    start = 0
    for expert in range(NUM_EXPERTS):
        end = tl.load(Offsets + expert)
        if GROUP_K:
            rows, reduction = M, end - start
            a_base = start * a_stride_k
            b_base = start * b_stride_k
            as_base = (start // 128 + expert) * as_stride_k
            bs_base = (start // 128 + expert) * bs_stride_k
            c_base = expert * M * N
        else:
            rows, reduction = end - start, K
            a_base = start * a_stride_m
            b_base = expert * b_stride_e
            as_base = start * as_stride_m
            bs_base = expert * bs_stride_e
            c_base = start * N

        tiles_n = tl.cdiv(N, BLOCK_N)
        last_tile = first_tile + tl.cdiv(rows, BLOCK_M) * tiles_n
        while tile < last_tile:
            local_tile = tile - first_tile
            row = (local_tile // tiles_n) * BLOCK_M + tl.arange(0, BLOCK_M)
            col = (local_tile % tiles_n) * BLOCK_N + tl.arange(0, BLOCK_N)
            inner = tl.arange(0, 128)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for block in range(tl.cdiv(reduction, 128)):
                k = block * 128 + inner
                a = tl.load(
                    A + a_base + row[:, None] * a_stride_m + k[None, :] * a_stride_k,
                    mask=(row[:, None] < rows) & (k[None, :] < reduction), other=0.0,
                )
                b = tl.load(
                    B + b_base + k[:, None] * b_stride_k + col[None, :] * b_stride_n,
                    mask=(k[:, None] < reduction) & (col[None, :] < N), other=0.0,
                )
                a_scale = tl.load(
                    AScale + as_base + row * as_stride_m + block * as_stride_k,
                    mask=row < rows, other=0.0,
                )
                scale_col = col if GROUP_K else col // 128
                b_scale = tl.load(
                    BScale + bs_base + block * bs_stride_k + scale_col * bs_stride_n,
                    mask=col < N, other=0.0,
                )
                # rescale each 128-term partial sum before accumulating in fp32
                partial = tl.dot(a, b, max_num_imprecise_acc=0)
                acc += partial * a_scale[:, None] * b_scale[None, :]
            tl.store(
                C + c_base + row[:, None] * N + col[None, :], acc,
                mask=(row[:, None] < rows) & (col[None, :] < N),
            )
            tile += NUM_PROGRAMS
        first_tile = last_tile
        start = end


def _quantize_activations(
    x: Tensor, offsets: Tensor, save_column: bool, deepgemm: bool = False, write_offsets: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    m, k = x.shape
    scale_rows = triton.cdiv(m, 128) + offsets.numel()
    row = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((m, triton.cdiv(k, 128)), device=x.device, dtype=torch.float32)
    column_m = scale_rows * 128 if deepgemm else m
    column = torch.empty((k, column_m) if save_column else (0,), device=x.device, dtype=torch.float8_e4m3fn)
    column_scale_rows = triton.cdiv(m, 512) + offsets.numel() if deepgemm else scale_rows
    column_scale = torch.empty(
        (column_scale_rows, k) if save_column else (0,), device=x.device,
        dtype=torch.int32 if deepgemm else torch.float32,
    )
    padded_offsets = torch.empty(offsets.shape if write_offsets else (0,), device=x.device, dtype=torch.int32)
    wrap_triton(_quantize_activations_kernel)[(scale_rows, triton.cdiv(k, 128))](
        x, row, column, row_scale, column_scale, offsets, padded_offsets,
        m, k, offsets.numel(), *x.stride(), save_column, deepgemm, column_m, write_offsets, num_warps=8,
    )
    return row, row_scale, column, column_scale, padded_offsets


def _grouped_fp8_mm(a: Tensor, b: Tensor, a_scale: Tensor, b_scale: Tensor, offsets: Tensor, dtype: torch.dtype) -> Tensor:
    group_k = b.ndim == 2
    m, k = a.shape
    n = b.shape[-1]
    num_experts = offsets.numel()
    out = torch.empty((num_experts, m, n) if group_k else (m, n), device=a.device, dtype=dtype)
    num_programs = torch.cuda.get_device_properties(a.device).multi_processor_count
    wrap_triton(_grouped_fp8_mm_kernel)[(num_programs,)](
        a, b, a_scale, b_scale, out, offsets, m, n, k,
        *a.stride(), 0 if group_k else b.stride(0), *b.stride()[-2:],
        *a_scale.stride(), 0 if group_k else b_scale.stride(0), *b_scale.stride()[-2:],
        num_experts, num_programs, group_k,
    )
    return out


@triton_op("gpt_moe::grouped_linear_fp8_forward", mutates_args=())
def _forward(
    x: Tensor, weight: Tensor, offsets: Tensor, save_backward: bool, deepgemm: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, row_scale, column, column_scale, _ = _quantize_activations(x, offsets, save_backward, deepgemm)
    experts, n, k = weight.shape
    weight_fp8 = torch.empty(weight.shape, device=weight.device, dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty((experts, triton.cdiv(n, 128), triton.cdiv(k, 128)), device=weight.device, dtype=torch.float32)
    wrap_triton(_quantize_weights_kernel)[(experts, triton.cdiv(n, 128), triton.cdiv(k, 128))](
        weight, weight_fp8, weight_scale, n, k, *weight.stride(), num_warps=8,
    )
    out = _grouped_fp8_mm(
        row, weight_fp8.transpose(-2, -1), row_scale, weight_scale.transpose(-2, -1), offsets, torch.bfloat16,
    )
    return out, column, column_scale, weight_fp8, weight_scale


@triton_op("gpt_moe::grouped_linear_fp8_backward", mutates_args=())
def _backward_gemms(
    grad: Tensor, x_column: Tensor, x_scale: Tensor, weight: Tensor, weight_scale: Tensor,
    offsets: Tensor, input_dtype: torch.dtype, weight_dtype: torch.dtype,
    deepgemm: bool,
) -> Tuple[Tensor, Tensor]:
    row, row_scale, column, column_scale, padded_offsets = _quantize_activations(grad, offsets, True, deepgemm, deepgemm)
    grad_x = _grouped_fp8_mm(row, weight, row_scale, weight_scale, offsets, input_dtype)
    if deepgemm:
        grad_weight = grouped_fp8_wgrad(column, column_scale, x_column, x_scale, padded_offsets, weight_dtype)
    else:
        grad_weight = _grouped_fp8_mm(column, x_column.t(), column_scale.t(), x_scale, offsets, weight_dtype)
    return grad_x, grad_weight


def _setup_context(ctx: Any, inputs: Tuple, output: Tuple) -> None:
    x, weight, offsets, _, ctx.deepgemm = inputs
    ctx.input_dtype = x.dtype
    ctx.weight_dtype = weight.dtype
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(*output[1:], offsets)


def _backward(ctx: Any, grad: Tensor, *unused: Tensor) -> Tuple:
    x_column, x_scale, weight, weight_scale, offsets = ctx.saved_tensors
    grad_x, grad_weight = _backward_gemms(
        grad, x_column, x_scale, weight, weight_scale, offsets, ctx.input_dtype, ctx.weight_dtype, ctx.deepgemm,
    )
    return grad_x, grad_weight, None, None, None


_forward.register_autograd(_backward, setup_context=_setup_context)


def grouped_linear_fp8(x: Tensor, weight: Tensor, offsets: Tensor, wgrad_backend: str = "triton") -> Tensor:
    if wgrad_backend not in ("triton", "deepgemm"):
        raise ValueError(f"unsupported grouped FP8 weight-gradient backend: {wgrad_backend}")
    save_backward = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    return _forward(x, weight, offsets, save_backward, wgrad_backend == "deepgemm")[0]
