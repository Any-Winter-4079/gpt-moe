from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .deepgemm import grouped_fp8_wgrad
from .dense import _matmul, _quantize_activations, quantize_weights
from .quantization import _inverse_scale


@triton.jit
def _quantize_output_kernel(
    X, Row, RowScale, Column, ColumnScale,
    M: tl.constexpr, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    xb: tl.constexpr, xh: tl.constexpr, xt: tl.constexpr, xd: tl.constexpr,
):
    tile = tl.program_id(0)
    block = tl.program_id(1)
    row = (tile * 128 + tl.arange(0, 128)).to(tl.int64)
    col = (block * 128 + tl.arange(0, 128)).to(tl.int64)
    width: tl.constexpr = H * D
    padded_width: tl.constexpr = tl.cdiv(width, 128) * 128
    # read the attention layout directly without materializing a BF16 token-major copy
    source = X + (row[:, None] // T) * xb + (col[None, :] // D) * xh
    source += (row[:, None] % T) * xt + (col[None, :] % D) * xd
    values = tl.load(source, (row[:, None] < M) & (col[None, :] < width), 0.0).to(tl.float32)
    row_scale = _inverse_scale(tl.max(tl.abs(values), axis=1))
    tl.store(Row + row[:, None] * padded_width + col[None, :], (values / row_scale[:, None]).to(tl.float8e4nv), row[:, None] < M)
    scale_m: tl.constexpr = tl.cdiv(M, 4) * 4
    row_scale_ptr = RowScale.to(tl.pointer_type(tl.uint8)) + ((block // 4) * scale_m + row) * 4
    row_exponent = (row_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
    tl.store(row_scale_ptr + block % 4, tl.where(row < M, row_exponent, 0), row < scale_m)
    if block == padded_width // 128 - 1:
        for padding in tl.static_range(1, 4):
            tl.store(row_scale_ptr + block % 4 + padding, 0, (row < scale_m) & (block % 4 + padding < 4))
    column_scale = _inverse_scale(tl.max(tl.abs(values), axis=0))
    column_m: tl.constexpr = tl.cdiv(M, 128) * 128
    tl.store(Column + col[None, :] * column_m + row[:, None], (values / column_scale[None, :]).to(tl.float8e4nv))
    scale_ptr = ColumnScale.to(tl.pointer_type(tl.uint8)) + ((tile // 4) * padded_width + col) * 4
    exponent = (column_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
    tl.store(scale_ptr + tile % 4, exponent)
    if tile == tl.cdiv(M, 128) - 1:
        for padding in tl.static_range(1, 4):
            tl.store(scale_ptr + tile % 4 + padding, 0, tile % 4 + padding < 4)


@triton_op("gpt_moe::attention_quantize_output", mutates_args=())
def _quantize_output(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, heads, seq_len, head_size = x.shape
    m, width = batch * seq_len, heads * head_size
    padded_width = triton.cdiv(width, 128) * 128
    row = torch.empty((m, padded_width), device=x.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((triton.cdiv(width, 512), triton.cdiv(m, 4) * 4), device=x.device, dtype=torch.int32)[:, :m].t()
    column = torch.empty((padded_width, triton.cdiv(m, 128) * 128), device=x.device, dtype=row.dtype)
    column_scale = torch.empty((triton.cdiv(m, 512), padded_width), device=x.device, dtype=torch.int32)
    wrap_triton(_quantize_output_kernel)[(triton.cdiv(m, 128), triton.cdiv(width, 128))](
        x, row, row_scale, column, column_scale, m, seq_len, heads, head_size,
        *x.stride(), num_warps=8,
    )
    return row, row_scale, column, column_scale


class _OutputProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, weight_scale, transposed, transposed_scale, parameter):
        row, row_scale, column, column_scale = _quantize_output(x)
        out = _matmul(row, weight, row_scale, weight_scale, torch.bfloat16)
        ctx.save_for_backward(column, column_scale, transposed, transposed_scale)
        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.weight_shape = parameter.shape
        ctx.weight_dtype = parameter.dtype
        return out[:, :parameter.shape[0]].reshape(x.shape[0], x.shape[2], parameter.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x_column, x_scale, transposed, transposed_scale = ctx.saved_tensors
        batch, heads, seq_len, head_size = ctx.input_shape
        gradient = grad.reshape(-1, ctx.weight_shape[0])
        row, row_scale, column, column_scale = _quantize_activations([gradient], True)
        grad_x = _matmul(row, transposed, row_scale, transposed_scale, ctx.input_dtype)
        offsets = torch.full((1,), gradient.shape[0], device=grad.device, dtype=torch.int32)
        grad_weight = grouped_fp8_wgrad(column, column_scale, x_column, x_scale, offsets, ctx.weight_dtype)[0]
        grad_x = grad_x[:, :heads * head_size].reshape(batch, seq_len, heads, head_size).transpose(1, 2)
        return grad_x, None, None, None, None, grad_weight[:ctx.weight_shape[0], :heads * head_size]


def project_output(
    x: Tensor, weight: Tensor, prepared_weight: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
) -> Tensor:
    if prepared_weight is None:
        with torch.no_grad():
            prepared_weight = quantize_weights([weight])
    return _OutputProjection.apply(x, *prepared_weight, weight)
