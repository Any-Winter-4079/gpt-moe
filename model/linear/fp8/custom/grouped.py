from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import custom_op, triton_op, wrap_triton

from .deepgemm import grouped_fp8_wgrad
from .quantization import _inverse_scale


@triton.jit
def _quantize_activations_kernel(
    X, Row, Column, RowScale, ColumnScale, Offsets, PaddedOffsets, Order,
    M: tl.constexpr, K: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    stride_m: tl.constexpr, stride_k: tl.constexpr, SAVE_COLUMN: tl.constexpr,
    DEEPGEMM: tl.constexpr, COLUMN_M: tl.constexpr, WRITE_OFFSETS: tl.constexpr,
    RELU2: tl.constexpr, PACK: tl.constexpr, TOP_K: tl.constexpr,
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
            if PACK:
                source_row = tl.load(Order + start + row, mask=start + row < end, other=0) // TOP_K
            else:
                source_row = start + row
            values = tl.load(
                X + source_row[:, None] * stride_m + col[None, :] * stride_k,
                mask=mask, other=0.0,
            ).to(tl.float32)
            if RELU2:
                # preserve the bf16 activation boundary before finding either scale
                positive = tl.maximum(values, 0.0)
                values = (positive * positive).to(X.dtype.element_ty).to(tl.float32)
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
def _quantize_expert_weights_kernel(
    Weights, Quantized, Scale,
    N: tl.constexpr, K: tl.constexpr, Strides: tl.constexpr,
):
    expert = tl.program_id(0)
    row = tl.program_id(1) * 128 + tl.arange(0, 128)
    col = tl.program_id(2) * 128 + tl.arange(0, 128)
    mask = (row[:, None] < N) & (col[None, :] < K)
    for index in tl.static_range(len(Weights)):
        if expert == index:
            values = tl.load(
                Weights[index] + row[:, None] * Strides[index][0] + col[None, :] * Strides[index][1],
                mask=mask, other=0.0,
            ).to(tl.float32)
            scale = _inverse_scale(tl.max(tl.max(tl.abs(values), axis=1), axis=0))
            quantized = (values / scale).to(tl.float8e4nv)
            tl.store(Quantized + expert * N * K + row[:, None] * K + col[None, :], quantized, mask=mask)
            tl.store(
                Scale + (expert * tl.cdiv(N, 128) + tl.program_id(1)) * tl.cdiv(K, 128) + tl.program_id(2),
                scale,
            )


@custom_op("gpt_moe::quantize_expert_weights", mutates_args=())
def _quantize_expert_weights(weights: List[Tensor]) -> Tuple[Tensor, Tensor]:
    n, k = weights[0].shape
    quantized = torch.empty((len(weights), n, k), device=weights[0].device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (len(weights), triton.cdiv(n, 128), triton.cdiv(k, 128)), device=weights[0].device, dtype=torch.float32,
    )
    # keep the pointer tuple inside the opaque op, outside wrap_triton's graph arguments
    _quantize_expert_weights_kernel[(len(weights), triton.cdiv(n, 128), triton.cdiv(k, 128))](
        tuple(weights), quantized, scales, n, k, tuple(weight.stride() for weight in weights), num_warps=8,
    )
    return quantized, scales


@_quantize_expert_weights.register_fake
def _fake_quantize_expert_weights(weights: List[Tensor]) -> Tuple[Tensor, Tensor]:
    n, k = weights[0].shape
    return (
        torch.empty((len(weights), n, k), device=weights[0].device, dtype=torch.float8_e4m3fn),
        torch.empty(
            (len(weights), triton.cdiv(n, 128), triton.cdiv(k, 128)), device=weights[0].device, dtype=torch.float32,
        ),
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=3, num_warps=warps)
        for bm, bn in [(64, 64), (64, 128), (128, 64)]
        for warps in (4, 8)
    ],
    key=["M", "N", "K", "NUM_EXPERTS", "GROUP_K", "RELU2_GRAD", "a_stride_m", "a_stride_k", "b_stride_k", "b_stride_n", "relu_stride_m", "relu_stride_n"],
)
@triton.jit
def _grouped_fp8_mm_kernel(
    A, B, AScale, BScale, C, Offsets, Relu2Input,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    a_stride_m: tl.constexpr, a_stride_k: tl.constexpr,
    b_stride_e: tl.constexpr, b_stride_k: tl.constexpr, b_stride_n: tl.constexpr,
    as_stride_m: tl.constexpr, as_stride_k: tl.constexpr,
    bs_stride_e: tl.constexpr, bs_stride_k: tl.constexpr, bs_stride_n: tl.constexpr,
    relu_stride_m: tl.constexpr, relu_stride_n: tl.constexpr,
    NUM_EXPERTS: tl.constexpr, NUM_PROGRAMS: tl.constexpr, GROUP_K: tl.constexpr,
    RELU2_GRAD: tl.constexpr,
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
                # retain 128-element scales while using four native 32-element scale groups
                a_exponent = (a_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
                b_exponent = (b_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
                a_exponents = tl.broadcast_to(a_exponent[:, None], (BLOCK_M, 4))
                b_exponents = tl.broadcast_to(b_exponent[:, None], (BLOCK_N, 4))
                acc = tl.dot_scaled(a, a_exponents, "e4m3", b, b_exponents, "e4m3", acc)
            if RELU2_GRAD:
                preactivation = tl.load(
                    Relu2Input + (start + row[:, None]) * relu_stride_m + col[None, :] * relu_stride_n,
                    mask=(row[:, None] < rows) & (col[None, :] < N), other=0.0,
                ).to(tl.float32)
                # round dgrad and the square derivative before their bf16 product
                rounded_grad = acc.to(C.dtype.element_ty).to(tl.float32)
                slope = (2.0 * tl.maximum(preactivation, 0.0)).to(C.dtype.element_ty).to(tl.float32)
                acc = tl.where(preactivation <= 0.0, 0.0, rounded_grad * slope)
            tl.store(
                C + c_base + row[:, None] * N + col[None, :], acc,
                mask=(row[:, None] < rows) & (col[None, :] < N),
            )
            tile += NUM_PROGRAMS
        first_tile = last_tile
        start = end


def _quantize_activations(
    x: Tensor, offsets: Tensor, save_column: bool, write_offsets: bool = False,
    relu2: bool = False,
    order: Optional[Tensor] = None, top_k: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    m, k = x.shape
    if order is not None:
        m = order.numel()
    scale_rows = triton.cdiv(m, 128) + offsets.numel()
    row = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((m, triton.cdiv(k, 128)), device=x.device, dtype=torch.float32)
    column_m = scale_rows * 128
    column = torch.empty((k, column_m) if save_column else (0,), device=x.device, dtype=torch.float8_e4m3fn)
    column_scale_rows = triton.cdiv(m, 512) + offsets.numel()
    column_scale = torch.empty(
        (column_scale_rows, k) if save_column else (0,), device=x.device,
        dtype=torch.int32,
    )
    padded_offsets = torch.empty(offsets.shape if write_offsets else (0,), device=x.device, dtype=torch.int32)
    wrap_triton(_quantize_activations_kernel)[(scale_rows, triton.cdiv(k, 128))](
        x, row, column, row_scale, column_scale, offsets, padded_offsets, x if order is None else order,
        m, k, offsets.numel(), *x.stride(), save_column, True, column_m, write_offsets,
        relu2, order is not None, top_k, num_warps=8,
    )
    return row, row_scale, column, column_scale, padded_offsets


def _grouped_fp8_mm(
    a: Tensor, b: Tensor, a_scale: Tensor, b_scale: Tensor, offsets: Tensor, dtype: torch.dtype,
    relu2_input: Optional[Tensor] = None,
) -> Tensor:
    if torch.cuda.get_device_capability(a.device) != (12, 0):
        raise ValueError("native scaled FP8 grouped projections require SM120")
    group_k = b.ndim == 2
    m, k = a.shape
    n = b.shape[-1]
    num_experts = offsets.numel()
    out = torch.empty((num_experts, m, n) if group_k else (m, n), device=a.device, dtype=dtype)
    num_programs = torch.cuda.get_device_properties(a.device).multi_processor_count
    wrap_triton(_grouped_fp8_mm_kernel)[(num_programs,)](
        a, b, a_scale, b_scale, out, offsets, a if relu2_input is None else relu2_input, m, n, k,
        *a.stride(), 0 if group_k else b.stride(0), *b.stride()[-2:],
        *a_scale.stride(), 0 if group_k else b_scale.stride(0), *b_scale.stride()[-2:],
        *(relu2_input.stride() if relu2_input is not None else (0, 0)),
        num_experts, num_programs, group_k, relu2_input is not None,
    )
    return out


@triton_op("gpt_moe::grouped_linear_fp8_weights_forward", mutates_args=())
def _weights_forward(
    x: Tensor, weights: List[Tensor], offsets: Tensor, save_backward: bool, relu2: bool,
    order: Optional[Tensor], top_k: int, inverse_order: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, row_scale, column, column_scale, _ = _quantize_activations(
        x, offsets, save_backward, relu2=relu2, order=order, top_k=top_k,
    )
    weight_fp8, weight_scale = _quantize_expert_weights(weights)
    out = _grouped_fp8_mm(
        row, weight_fp8.transpose(-2, -1), row_scale, weight_scale.transpose(-2, -1), offsets, torch.bfloat16,
    )
    return out, column, column_scale, weight_fp8, weight_scale


@triton_op("gpt_moe::grouped_linear_fp8_cached_weights_forward", mutates_args=())
def _cached_weights_forward(
    x: Tensor, weights: List[Tensor], weight_fp8: Tensor, weight_scale: Tensor, offsets: Tensor,
    save_backward: bool, relu2: bool, order: Optional[Tensor], top_k: int, inverse_order: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    row, row_scale, column, column_scale, _ = _quantize_activations(
        x, offsets, save_backward, relu2=relu2, order=order, top_k=top_k,
    )
    out = _grouped_fp8_mm(
        row, weight_fp8.transpose(-2, -1), row_scale, weight_scale.transpose(-2, -1), offsets, torch.bfloat16,
    )
    return out, column, column_scale


@triton_op("gpt_moe::grouped_linear_fp8_backward", mutates_args=())
def _backward_gemms(
    grad: Tensor, x_column: Tensor, x_scale: Tensor, weight: Tensor, weight_scale: Tensor,
    offsets: Tensor, input_dtype: torch.dtype, weight_dtype: torch.dtype,
    relu2_input: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    row, row_scale, column, column_scale, padded_offsets = _quantize_activations(grad, offsets, True, write_offsets=True)
    grad_x = _grouped_fp8_mm(row, weight, row_scale, weight_scale, offsets, input_dtype, relu2_input)
    grad_weight = grouped_fp8_wgrad(column, column_scale, x_column, x_scale, padded_offsets, weight_dtype)
    return grad_x, grad_weight


@triton.jit
def _unpack_input_gradient_kernel(
    Packed, InverseOrder, Output,
    M: tl.constexpr, K: tl.constexpr, TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (row[:, None] < M) & (col[None, :] < K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for selected in tl.static_range(TOP_K):
        packed_row = tl.load(InverseOrder + row * TOP_K + selected, mask=row < M, other=0)
        values = tl.load(Packed + packed_row[:, None] * K + col[None, :], mask=mask, other=0.0)
        # retain the destination dtype after each expert contribution
        acc = (acc + values.to(tl.float32)).to(Output.dtype.element_ty).to(tl.float32)
    tl.store(Output + row[:, None] * K + col[None, :], acc, mask=mask)


@triton_op("gpt_moe::unpack_fp8_input_gradient", mutates_args=())
def _unpack_input_gradient(packed: Tensor, inverse_order: Tensor, top_k: int) -> Tensor:
    m, k = inverse_order.numel() // top_k, packed.shape[1]
    output = packed.new_empty((m, k))
    wrap_triton(_unpack_input_gradient_kernel)[(triton.cdiv(m, 16), triton.cdiv(k, 128))](
        packed, inverse_order, output, m, k, top_k, 16, 128,
    )
    return output


def _setup_weights_context(ctx: Any, inputs: Tuple, output: Tuple) -> None:
    x, weights, offsets, _, relu2, _, ctx.top_k, inverse_order = inputs
    ctx.input_shape = x.shape
    ctx.input_dtype = x.dtype
    ctx.weight_dtype = weights[0].dtype
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(*output[1:], offsets, x if relu2 else None, inverse_order)


def _weights_backward(ctx: Any, grad: Tensor, *unused: Tensor) -> Tuple:
    x_column, x_scale, weight, weight_scale, offsets, preactivation, inverse_order = ctx.saved_tensors
    grad_x, grad_weight = _backward_gemms(
        grad, x_column, x_scale, weight, weight_scale, offsets, ctx.input_dtype, ctx.weight_dtype,
        preactivation,
    )
    if inverse_order is not None:
        grad_x = _unpack_input_gradient(grad_x, inverse_order, ctx.top_k)
    return grad_x, list(grad_weight.unbind(0)), None, None, None, None, None, None


_weights_forward.register_autograd(_weights_backward, setup_context=_setup_weights_context)


def _setup_cached_weights_context(ctx: Any, inputs: Tuple, output: Tuple) -> None:
    x, weights, weight_fp8, weight_scale, offsets, _, relu2, _, ctx.top_k, inverse_order = inputs
    ctx.input_shape = x.shape
    ctx.input_dtype = x.dtype
    ctx.weight_dtype = weights[0].dtype
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(*output[1:], weight_fp8, weight_scale, offsets, x if relu2 else None, inverse_order)


def _cached_weights_backward(ctx: Any, grad: Tensor, *unused: Tensor) -> Tuple:
    grad_x, grad_weights, *_ = _weights_backward(ctx, grad, *unused)
    return grad_x, grad_weights, None, None, None, None, None, None, None, None


_cached_weights_forward.register_autograd(_cached_weights_backward, setup_context=_setup_cached_weights_context)


def grouped_linear_fp8_weights(
    x: Tensor, weights: List[Tensor], offsets: Tensor, relu2: bool = False,
    order: Optional[Tensor] = None, top_k: int = 1,
    prepared_weight: Optional[Tuple[Tensor, Tensor]] = None,
    inverse_order: Optional[Tensor] = None,
) -> Tensor:
    if relu2 and x.dtype != torch.bfloat16:
        raise ValueError("fused grouped FP8 ReLU2 requires bfloat16 preactivations")
    if order is not None and inverse_order is None:
        inverse_order = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=order.device))
    save_backward = torch.is_grad_enabled() and (x.requires_grad or any(weight.requires_grad for weight in weights))
    if prepared_weight is not None:
        return _cached_weights_forward(
            x, weights, *prepared_weight, offsets, save_backward, relu2, order, top_k, inverse_order,
        )[0]
    return _weights_forward(x, weights, offsets, save_backward, relu2, order, top_k, inverse_order)[0]

