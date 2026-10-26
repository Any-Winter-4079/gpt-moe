from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .quantization import _inverse_scale


@triton.jit
def _gain_kernel(X, Gain, M: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row[:, None] * N + col[None, :], (row[:, None] < M) & (col[None, :] < N), 0.0)
    gain = tl.rsqrt(tl.sum(x * x, axis=1) / N + EPS)
    tl.store(Gain + row, gain, row < M)


@triton.jit
def _quantize_kernel(
    X, Weight, Gain, Row, RowScale, Column, ColumnScale, M: tl.constexpr, N: tl.constexpr,
):
    tile = tl.program_id(0)
    block = tl.program_id(1)
    row = tile * 128 + tl.arange(0, 128)
    col = block * 128 + tl.arange(0, 128)
    valid = (row[:, None] < M) & (col[None, :] < N)
    x = tl.load(X + row[:, None] * N + col[None, :], valid, 0.0)
    weight = tl.load(Weight + col, col < N, 0.0)
    gain = tl.load(Gain + row, row < M, 0.0)
    # retain the FP32 RMSNorm boundary before both FP8 scale reductions
    values = (x * gain[:, None]) * weight[None, :]
    width: tl.constexpr = tl.cdiv(N, 128) * 128
    row_scale = _inverse_scale(tl.max(tl.abs(values), axis=1))
    tl.store(Row + row[:, None] * width + col[None, :], (values / row_scale[:, None]).to(tl.float8e4nv), row[:, None] < M)
    scale_m: tl.constexpr = tl.cdiv(M, 4) * 4
    row_ptr = RowScale.to(tl.pointer_type(tl.uint8)) + ((block // 4) * scale_m + row) * 4
    exponent = (row_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
    tl.store(row_ptr + block % 4, tl.where(row < M, exponent, 0), row < scale_m)
    if block == width // 128 - 1:
        for padding in tl.static_range(1, 4):
            tl.store(row_ptr + block % 4 + padding, 0, (row < scale_m) & (block % 4 + padding < 4))
    column_scale = _inverse_scale(tl.max(tl.abs(values), axis=0))
    column_m: tl.constexpr = tl.cdiv(M, 128) * 128
    tl.store(Column + col[None, :] * column_m + row[:, None], (values / column_scale[None, :]).to(tl.float8e4nv))
    column_ptr = ColumnScale.to(tl.pointer_type(tl.uint8)) + ((tile // 4) * width + col) * 4
    exponent = (column_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
    tl.store(column_ptr + tile % 4, exponent)
    if tile == tl.cdiv(M, 128) - 1:
        for padding in tl.static_range(1, 4):
            tl.store(column_ptr + tile % 4 + padding, 0, tile % 4 + padding < 4)


@triton_op("gpt_moe::rms_norm_quantize", mutates_args=())
def quantize(x: Tensor, weight: Tensor, eps: float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    m, n = x.shape
    width = triton.cdiv(n, 128) * 128
    gain = torch.empty((m,), device=x.device, dtype=torch.float32)
    row = torch.empty((m, width), device=x.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((triton.cdiv(n, 512), triton.cdiv(m, 4) * 4), device=x.device, dtype=torch.int32)[:, :m].t()
    column = torch.empty((width, triton.cdiv(m, 128) * 128), device=x.device, dtype=row.dtype)
    column_scale = torch.empty((triton.cdiv(m, 512), width), device=x.device, dtype=torch.int32)
    wrap_triton(_gain_kernel)[(triton.cdiv(m, 4),)](
        x, gain, m, n, eps, triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False,
    )
    wrap_triton(_quantize_kernel)[(triton.cdiv(m, 128), width // 128)](
        x, weight, gain, row, row_scale, column, column_scale, m, n, num_warps=8, enable_fp_fusion=False,
    )
    return row, row_scale, column, column_scale, gain


@triton.jit
def _backward_kernel(X, Weight, Gain, DY, DX, DWeightPart, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    row = tile * 16 + tl.arange(0, 16)
    col = tl.arange(0, BLOCK)
    mask = (row[:, None] < M) & (col[None, :] < N)
    x = tl.load(X + row[:, None] * N + col[None, :], mask, 0.0)
    dy = tl.load(DY + row[:, None] * N + col[None, :], mask, 0.0)
    weight = tl.load(Weight + col, col < N, 0.0)
    gain = tl.load(Gain + row, row < M, 0.0)
    weighted_grad = dy * weight[None, :]
    correlation = tl.sum(weighted_grad * x, axis=1) / N
    dx = gain[:, None] * (weighted_grad - x * ((gain * gain) * correlation)[:, None])
    tl.store(DX + row[:, None] * N + col[None, :], dx, mask)
    partial = tl.sum(dy * (x * gain[:, None]), axis=0)
    tl.store(DWeightPart + tile * N + col, partial, col < N)


@triton_op("gpt_moe::rms_norm_quantize_backward", mutates_args=())
def backward(x: Tensor, weight: Tensor, gain: Tensor, grad: Tensor) -> Tuple[Tensor, Tensor]:
    m, n = x.shape
    dx = torch.empty_like(x)
    weight_part = torch.empty((triton.cdiv(m, 16), n), device=x.device, dtype=torch.float32)
    wrap_triton(_backward_kernel)[(triton.cdiv(m, 16),)](
        x, weight, gain, grad, dx, weight_part, m, n, triton.next_power_of_2(n),
        num_warps=8, enable_fp_fusion=False,
    )
    return dx, weight_part
