from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import custom_op, triton_op, wrap_triton

from .deepgemm import grouped_fp8_wgrad
from .quantization import _inverse_scale


@triton.jit
def _quantize_weights_kernel(
    W0, W1, W2, Quantized, Scale, Transposed, TransposedScale,
    N0: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, K: tl.constexpr,
    s0n: tl.constexpr, s0k: tl.constexpr, s1n: tl.constexpr, s1k: tl.constexpr,
    s2n: tl.constexpr, s2k: tl.constexpr,
):
    block = tl.program_id(0)
    row = tl.arange(0, 128)
    col = tl.program_id(1) * 128 + tl.arange(0, 128)
    if block < tl.cdiv(N0, 128):
        row += block * 128
        values = tl.load(W0 + row[:, None] * s0n + col[None, :] * s0k,
                         (row[:, None] < N0) & (col[None, :] < K), 0).to(tl.float32)
    elif block < tl.cdiv(N0, 128) + tl.cdiv(N1, 128):
        row += (block - tl.cdiv(N0, 128)) * 128
        values = tl.load(W1 + row[:, None] * s1n + col[None, :] * s1k,
                         (row[:, None] < N1) & (col[None, :] < K), 0).to(tl.float32)
    else:
        row += (block - tl.cdiv(N0, 128) - tl.cdiv(N1, 128)) * 128
        values = tl.load(W2 + row[:, None] * s2n + col[None, :] * s2k,
                         (row[:, None] < N2) & (col[None, :] < K), 0).to(tl.float32)
    scale = _inverse_scale(tl.max(tl.max(tl.abs(values), axis=1), axis=0))
    quantized = (values / scale).to(tl.float8e4nv)
    packed_row = block * 128 + tl.arange(0, 128)
    n: tl.constexpr = (tl.cdiv(N0, 128) + tl.cdiv(N1, 128) + tl.cdiv(N2, 128)) * 128
    k: tl.constexpr = tl.cdiv(K, 128) * 128
    tl.store(Quantized + packed_row[:, None] * k + col[None, :], quantized)
    tl.store(Transposed + col[None, :] * n + packed_row[:, None], quantized)
    exponent = (scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
    k_block = tl.program_id(1)
    scale_ptr = Scale.to(tl.pointer_type(tl.uint8)) + ((k_block // 4) * n + packed_row) * 4
    transposed_ptr = TransposedScale.to(tl.pointer_type(tl.uint8)) + ((block // 4) * k + col) * 4
    tl.store(scale_ptr + k_block % 4, exponent)
    tl.store(transposed_ptr + block % 4, exponent)
    for padding in tl.static_range(1, 4):
        if k_block == k // 128 - 1:
            tl.store(scale_ptr + k_block % 4 + padding, 0, k_block % 4 + padding < 4)
        if block == n // 128 - 1:
            tl.store(transposed_ptr + block % 4 + padding, 0, block % 4 + padding < 4)


@triton_op("gpt_moe::attention_quantize_weights", mutates_args=())
def quantize_weights(weights: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    widths = [weight.shape[0] for weight in weights]
    # keep 128-element scale blocks within each original projection
    n = sum(triton.cdiv(width, 128) * 128 for width in widths)
    k = weights[0].shape[1]
    padded_k = triton.cdiv(k, 128) * 128
    quantized = torch.empty((n, padded_k), device=weights[0].device, dtype=torch.float8_e4m3fn)
    transposed = torch.empty((padded_k, n), device=weights[0].device, dtype=quantized.dtype)
    # native dense GEMMs read MN-major scales with four UE8M0 bytes per int32
    scale = torch.empty((triton.cdiv(k, 512), n), device=weights[0].device, dtype=torch.int32).t()
    transposed_scale = torch.empty((triton.cdiv(n, 512), padded_k), device=weights[0].device, dtype=torch.int32).t()
    w0, w1, w2 = weights if len(weights) == 3 else (weights[0], weights[0], weights[0])
    n0, n1, n2 = widths if len(weights) == 3 else (widths[0], 0, 0)
    wrap_triton(_quantize_weights_kernel)[(n // 128, triton.cdiv(k, 128))](
        w0, w1, w2, quantized, scale, transposed, transposed_scale,
        n0, n1, n2, k, *w0.stride(), *w1.stride(), *w2.stride(), num_warps=8,
    )
    return quantized, scale, transposed, transposed_scale


@triton.jit
def _quantize_activations_kernel(
    X0, X1, X2, Row, RowScale, Column, ColumnScale,
    M: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
    s0m: tl.constexpr, s0k: tl.constexpr, s1m: tl.constexpr, s1k: tl.constexpr,
    s2m: tl.constexpr, s2k: tl.constexpr, SAVE_COLUMN: tl.constexpr,
):
    tile = tl.program_id(0)
    block = tl.program_id(1)
    row = tile * 128 + tl.arange(0, 128)
    col = tl.arange(0, 128)
    if block < tl.cdiv(K0, 128):
        col += block * 128
        values = tl.load(X0 + row[:, None] * s0m + col[None, :] * s0k,
                         (row[:, None] < M) & (col[None, :] < K0), 0).to(tl.float32)
    elif block < tl.cdiv(K0, 128) + tl.cdiv(K1, 128):
        col += (block - tl.cdiv(K0, 128)) * 128
        values = tl.load(X1 + row[:, None] * s1m + col[None, :] * s1k,
                         (row[:, None] < M) & (col[None, :] < K1), 0).to(tl.float32)
    else:
        col += (block - tl.cdiv(K0, 128) - tl.cdiv(K1, 128)) * 128
        values = tl.load(X2 + row[:, None] * s2m + col[None, :] * s2k,
                         (row[:, None] < M) & (col[None, :] < K2), 0).to(tl.float32)
    K: tl.constexpr = (tl.cdiv(K0, 128) + tl.cdiv(K1, 128) + tl.cdiv(K2, 128)) * 128
    packed_col = block * 128 + tl.arange(0, 128)
    row_scale = _inverse_scale(tl.max(tl.abs(values), axis=1))
    tl.store(Row + row[:, None] * K + packed_col[None, :], (values / row_scale[:, None]).to(tl.float8e4nv), row[:, None] < M)
    scale_m: tl.constexpr = tl.cdiv(M, 4) * 4
    row_scale_ptr = RowScale.to(tl.pointer_type(tl.uint8)) + ((block // 4) * scale_m + row) * 4
    row_exponent = (row_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
    tl.store(row_scale_ptr + block % 4, tl.where(row < M, row_exponent, 0), row < scale_m)
    if block == K // 128 - 1:
        for padding in tl.static_range(1, 4):
            tl.store(row_scale_ptr + block % 4 + padding, 0, (row < scale_m) & (block % 4 + padding < 4))
    if SAVE_COLUMN:
        column_scale = _inverse_scale(tl.max(tl.abs(values), axis=0))
        column_m: tl.constexpr = tl.cdiv(M, 128) * 128
        tl.store(Column + packed_col[None, :] * column_m + row[:, None], (values / column_scale[None, :]).to(tl.float8e4nv))
        scale_ptr = ColumnScale.to(tl.pointer_type(tl.uint8)) + ((tile // 4) * K + packed_col) * 4
        exponent = (column_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
        tl.store(scale_ptr + tile % 4, exponent)
        if tile == tl.cdiv(M, 128) - 1:
            for padding in tl.static_range(1, 4):
                tl.store(scale_ptr + tile % 4 + padding, 0, tile % 4 + padding < 4)


@triton_op("gpt_moe::attention_dg_quantize_activations", mutates_args=())
def _quantize_activations(inputs: List[Tensor], save_column: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    m = inputs[0].shape[0]
    widths = [value.shape[1] for value in inputs]
    k = sum(triton.cdiv(width, 128) * 128 for width in widths)
    row = torch.empty((m, k), device=inputs[0].device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((triton.cdiv(k, 512), triton.cdiv(m, 4) * 4), device=row.device, dtype=torch.int32)[:, :m].t()
    column = torch.empty((k, triton.cdiv(m, 128) * 128) if save_column else (0,), device=row.device, dtype=row.dtype)
    column_scale = torch.empty((triton.cdiv(m, 512), k) if save_column else (0,), device=row.device, dtype=torch.int32)
    x0, x1, x2 = inputs if len(inputs) == 3 else (inputs[0], inputs[0], inputs[0])
    k0, k1, k2 = widths if len(inputs) == 3 else (widths[0], 0, 0)
    wrap_triton(_quantize_activations_kernel)[(triton.cdiv(m, 128), k // 128)](
        x0, x1, x2, row, row_scale, column, column_scale, m, k0, k1, k2,
        *x0.stride(), *x1.stride(), *x2.stride(), save_column, num_warps=8,
    )
    return row, row_scale, column, column_scale


@custom_op("gpt_moe::attention_deepgemm_mm", mutates_args=())
def _matmul(a: Tensor, b: Tensor, a_scale: Tensor, b_scale: Tensor, dtype: torch.dtype) -> Tensor:
    import deep_gemm

    if torch.cuda.get_device_capability(a.device) != (12, 0):
        raise ValueError("DeepGEMM FP8 attention projections require SM120")
    out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=dtype)
    deep_gemm.fp8_gemm_nt((a, a_scale), (b, b_scale), out, recipe=(1, 1, 128), disable_ue8m0_cast=True)
    return out


@_matmul.register_fake
def _matmul_fake(a, b, a_scale, b_scale, dtype):
    return torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=dtype)


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, weight_scale, transposed, transposed_scale, *weights):
        row, row_scale, column, column_scale = _quantize_activations([x], True)
        out = _matmul(row, weight, row_scale, weight_scale, torch.bfloat16)
        ctx.save_for_backward(column, column_scale, transposed, transposed_scale)
        ctx.tokens = x.shape[0]
        ctx.input_width = x.shape[1]
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weights[0].dtype
        ctx.widths = [value.shape[0] for value in weights]
        ctx.set_materialize_grads(False)
        # return views here so backward receives Q/K/V gradients without a bf16 concatenation
        outputs, start = [], 0
        for width in ctx.widths:
            outputs.append(out[:, start:start + width])
            start += triton.cdiv(width, 128) * 128
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *gradients):
        x_column, x_scale, transposed, transposed_scale = ctx.saved_tensors
        gradients = [
            grad if grad is not None else torch.zeros((ctx.tokens, width), device=transposed.device, dtype=torch.bfloat16)
            for grad, width in zip(gradients, ctx.widths)
        ]
        row, row_scale, column, column_scale = _quantize_activations(gradients, True)
        grad_x = _matmul(row, transposed, row_scale, transposed_scale, ctx.input_dtype)
        offsets = torch.full((1,), gradients[0].shape[0], device=transposed.device, dtype=torch.int32)
        grad_weight = grouped_fp8_wgrad(column, column_scale, x_column, x_scale, offsets, ctx.weight_dtype)[0]
        weight_grads, start = [], 0
        for width in ctx.widths:
            weight_grads.append(grad_weight[start:start + width, :ctx.input_width])
            start += triton.cdiv(width, 128) * 128
        return (grad_x[:, :ctx.input_width], None, None, None, None, *weight_grads)


def project(
    x: Tensor, weights: List[Tensor], prepared_weight: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
) -> List[Tensor]:
    if prepared_weight is None:
        with torch.no_grad():
            prepared_weight = quantize_weights(weights)
    inputs = x.reshape(-1, x.shape[-1])
    if torch.is_grad_enabled() and (x.requires_grad or any(weight.requires_grad for weight in weights)):
        outputs = _Projection.apply(inputs, *prepared_weight, *weights)
    else:
        row, row_scale, _, _ = _quantize_activations([inputs], False)
        out = _matmul(row, prepared_weight[0], row_scale, prepared_weight[1], torch.bfloat16)
        outputs, start = [], 0
        for weight in weights:
            outputs.append(out[:, start:start + weight.shape[0]])
            start += triton.cdiv(weight.shape[0], 128) * 128
    return [value.reshape(*x.shape[:-1], weight.shape[0]) for value, weight in zip(outputs, weights)]
