from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from ..linear.fp8.custom.dense import _matmul, _quantize_activations, quantize_weights
from ..linear.fp8.custom.deepgemm import grouped_fp8_wgrad
from ..linear.fp8.custom.quantization import _inverse_scale


@triton.jit
def _rotate(x, paired, cos, sin, feature, dtype: tl.constexpr, inverse: tl.constexpr):
    first = (x * cos).to(dtype).to(tl.float32)
    second = (paired * sin).to(dtype).to(tl.float32)
    subtract = (feature % 2 == 1) if inverse else (feature % 2 == 0)
    return (first + tl.where(subtract, -second, second)).to(dtype).to(tl.float32)


@triton.jit
def _forward_kernel(
    Q, K, QScale, KScale, Cos, Sin, QOut, KOut, QGain, KGain,
    T: tl.constexpr, H: tl.constexpr,
    qb: tl.constexpr, qh: tl.constexpr, qt: tl.constexpr, qd: tl.constexpr,
    kb: tl.constexpr, kh: tl.constexpr, kt: tl.constexpr, kd: tl.constexpr,
    ct: tl.constexpr, cd: tl.constexpr, st: tl.constexpr, sd: tl.constexpr,
    EPS: tl.constexpr, RMS: tl.constexpr,
    SCALE_MIN: tl.constexpr, SCALE_MAX: tl.constexpr,
    MIN_INF: tl.constexpr, MAX_INF: tl.constexpr,
):
    scale_min: tl.constexpr = SCALE_MIN if MIN_INF == 0 else (float("-inf") if MIN_INF < 0 else float("inf"))
    scale_max: tl.constexpr = SCALE_MAX if MAX_INF == 0 else (float("-inf") if MAX_INF < 0 else float("inf"))
    batch_head = tl.program_id(0)
    batch = batch_head // (2 * H)
    head = batch_head % (2 * H)
    token = tl.program_id(1) * 16 + tl.arange(0, 16)
    feature = tl.arange(0, 64)
    valid = token[:, None] < T
    if head < H:
        source = Q + batch * qb + head * qh + token[:, None] * qt
        x = tl.load(source + feature[None, :] * qd, valid, 0.0).to(tl.float32)
        scale = tl.load(QScale + head).to(tl.float32)
        out = QOut + ((batch * H + head) * T + token[:, None]) * 64 + feature[None, :]
        gain_out = QGain + (batch * H + head) * T + token
    else:
        head -= H
        source = K + batch * kb + head * kh + token[:, None] * kt
        x = tl.load(source + feature[None, :] * kd, valid, 0.0).to(tl.float32)
        scale = tl.load(KScale + head).to(tl.float32)
        out = KOut + ((batch * H + head) * T + token[:, None]) * 64 + feature[None, :]
        gain_out = KGain + (batch * H + head) * T + token
    dtype: tl.constexpr = Q.dtype.element_ty
    paired = tl.gather(x, tl.broadcast_to(feature[None, :] ^ 1, (16, 64)), axis=1)
    cos = tl.load(Cos + token[:, None] * ct + (feature[None, :] // 2) * cd, valid, 0.0).to(dtype).to(tl.float32)
    sin = tl.load(Sin + token[:, None] * st + (feature[None, :] // 2) * sd, valid, 0.0).to(dtype).to(tl.float32)
    rotated = _rotate(x, paired, cos, sin, feature[None, :], dtype, False)
    # preserve CUDA BF16 autocast's FP32 norm statistic and rsqrt
    statistic = tl.sum(rotated * rotated, axis=1)
    if RMS:
        statistic /= 64
    gain = tl.rsqrt(statistic + EPS)
    scale = tl.where(scale != scale, scale, tl.minimum(tl.maximum(scale, scale_min), scale_max)).to(dtype).to(tl.float32)
    normalized = (rotated * gain[:, None].to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    tl.store(out, normalized * scale, valid)
    tl.store(gain_out, gain, token < T)


@triton_op("gpt_moe::attention_qkv_norm_rope_forward", mutates_args=())
def _forward(
    q: Tensor, k: Tensor, q_scale: Tensor, k_scale: Tensor, cos: Tensor, sin: Tensor,
    eps: float, rms: bool, scale_min: float, scale_max: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, heads, seq_len, _ = q.shape
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    q_gain = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
    k_gain = torch.empty_like(q_gain)
    # keep infinities out of Inductor's generated Python metadata
    min_inf = (scale_min == float("inf")) - (scale_min == float("-inf"))
    max_inf = (scale_max == float("inf")) - (scale_max == float("-inf"))
    wrap_triton(_forward_kernel)[(batch * 2 * heads, triton.cdiv(seq_len, 16))](
        q, k, q_scale, k_scale, cos, sin, q_out, k_out, q_gain, k_gain,
        seq_len, heads, *q.stride(), *k.stride(), *cos.stride()[-2:], *sin.stride()[-2:],
        eps, rms, 0.0 if min_inf else scale_min, 0.0 if max_inf else scale_max,
        min_inf, max_inf, num_warps=4, enable_fp_fusion=False,
    )
    return q_out, k_out, q_gain, k_gain


@triton.jit
def _backward_quantize_kernel(
    Q, K, QScale, KScale, Cos, Sin, QGain, KGain, DQOut, DKOut, DVOut,
    Row, RowScale, Column, ColumnScale, DScalePart,
    M: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    qb: tl.constexpr, qh: tl.constexpr, qt: tl.constexpr, qd: tl.constexpr,
    kb: tl.constexpr, kh: tl.constexpr, kt: tl.constexpr, kd: tl.constexpr,
    dqb: tl.constexpr, dqh: tl.constexpr, dqt: tl.constexpr, dqd: tl.constexpr,
    dkb: tl.constexpr, dkh: tl.constexpr, dkt: tl.constexpr, dkd: tl.constexpr,
    dvb: tl.constexpr, dvh: tl.constexpr, dvt: tl.constexpr, dvd: tl.constexpr,
    ct: tl.constexpr, cd: tl.constexpr, st: tl.constexpr, sd: tl.constexpr,
    RMS: tl.constexpr, SCALE_MIN: tl.constexpr, SCALE_MAX: tl.constexpr,
    MIN_INF: tl.constexpr, MAX_INF: tl.constexpr,
):
    scale_min: tl.constexpr = SCALE_MIN if MIN_INF == 0 else (float("-inf") if MIN_INF < 0 else float("inf"))
    scale_max: tl.constexpr = SCALE_MAX if MAX_INF == 0 else (float("-inf") if MAX_INF < 0 else float("inf"))
    tile = tl.program_id(0)
    block = tl.program_id(1)
    row = tile * 128 + tl.arange(0, 128)
    token = row % T
    batch = row // T
    feature = tl.arange(0, 128)
    projection_block = block % (H // 2)
    head = projection_block * 2 + feature // 64
    dim = feature % 64
    valid = row[:, None] < M
    dtype: tl.constexpr = Q.dtype.element_ty
    if block < H:
        head_pair = projection_block * 2 + tl.arange(0, 2)
        if block < H // 2:
            source = Q + batch[:, None] * qb + head[None, :] * qh + token[:, None] * qt
            x = tl.load(source + dim[None, :] * qd, valid, 0.0).to(tl.float32)
            scale = tl.load(QScale + head).to(tl.float32)
            head_gain = tl.load(QGain + (batch[:, None] * H + head_pair[None, :]) * T + token[:, None], valid, 0.0).to(tl.float32)
            grad = tl.load(DQOut + batch[:, None] * dqb + head[None, :] * dqh + token[:, None] * dqt + dim[None, :] * dqd, valid, 0.0).to(tl.float32)
            scale_head = head_pair
        else:
            source = K + batch[:, None] * kb + head[None, :] * kh + token[:, None] * kt
            x = tl.load(source + dim[None, :] * kd, valid, 0.0).to(tl.float32)
            scale = tl.load(KScale + head).to(tl.float32)
            head_gain = tl.load(KGain + (batch[:, None] * H + head_pair[None, :]) * T + token[:, None], valid, 0.0).to(tl.float32)
            grad = tl.load(DKOut + batch[:, None] * dkb + head[None, :] * dkh + token[:, None] * dkt + dim[None, :] * dkd, valid, 0.0).to(tl.float32)
            scale_head = H + head_pair
        gain = tl.broadcast_to(head_gain[:, :, None].to(dtype).to(tl.float32), (128, 2, 64)).reshape(128, 128)
        pair_index = tl.broadcast_to(feature[None, :] ^ 1, (128, 128))
        paired = tl.gather(x, pair_index, axis=1)
        cos = tl.load(Cos + token[:, None] * ct + (dim[None, :] // 2) * cd, valid, 0.0).to(dtype).to(tl.float32)
        sin = tl.load(Sin + token[:, None] * st + (dim[None, :] // 2) * sd, valid, 0.0).to(dtype).to(tl.float32)
        rotated = _rotate(x, paired, cos, sin, dim[None, :], dtype, False)
        normalized = (rotated * gain).to(dtype).to(tl.float32)
        scale_part = (grad * normalized).to(dtype).to(tl.float32).reshape(128, 2, 64)
        partial = tl.sum(tl.sum(scale_part, axis=2), axis=0)
        tl.store(DScalePart + scale_head * tl.cdiv(M, 128) + tile, partial)
        scale = tl.where(scale != scale, scale, tl.minimum(tl.maximum(scale, scale_min), scale_max)).to(dtype).to(tl.float32)
        grad = (grad * scale[None, :]).to(dtype).to(tl.float32)
        grad_gain = tl.sum((grad * rotated).to(dtype).to(tl.float32).reshape(128, 2, 64), axis=2).to(dtype).to(tl.float32)
        # rsqrt backward consumes the unrounded FP32 gain
        cube = (head_gain * head_gain) * head_gain
        grad_statistic = (grad_gain * -0.5) * cube
        if RMS:
            grad_statistic /= 64
        direct = (grad * gain).to(dtype).to(tl.float32)
        indirect = (grad_statistic[:, :, None] * (2.0 * rotated).reshape(128, 2, 64)).to(dtype).to(tl.float32)
        grad_rotated = (direct + indirect.reshape(128, 128)).to(dtype).to(tl.float32)
        paired_grad = tl.gather(grad_rotated, pair_index, axis=1)
        # reload the small rotary tables after the norm derivative to shorten register lifetimes
        cos = tl.load(Cos + token[:, None] * ct + (dim[None, :] // 2) * cd, valid, 0.0).to(dtype).to(tl.float32)
        sin = tl.load(Sin + token[:, None] * st + (dim[None, :] // 2) * sd, valid, 0.0).to(dtype).to(tl.float32)
        values = _rotate(grad_rotated, paired_grad, cos, sin, dim[None, :], dtype, True)
    else:
        values = tl.load(DVOut + batch[:, None] * dvb + head[None, :] * dvh + token[:, None] * dvt + dim[None, :] * dvd, valid, 0.0).to(tl.float32)
    values = tl.where(valid, values, 0.0)
    width: tl.constexpr = 3 * H * 64
    packed_col = block * 128 + feature
    row_scale = _inverse_scale(tl.max(tl.abs(values), axis=1))
    tl.store(Row + row[:, None] * width + packed_col[None, :], (values / row_scale[:, None]).to(tl.float8e4nv), valid)
    scale_m: tl.constexpr = tl.cdiv(M, 4) * 4
    row_scale_ptr = RowScale.to(tl.pointer_type(tl.uint8)) + ((block // 4) * scale_m + row) * 4
    row_exponent = (row_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
    tl.store(row_scale_ptr + block % 4, tl.where(row < M, row_exponent, 0), row < scale_m)
    if block == width // 128 - 1:
        for padding in tl.static_range(1, 4):
            tl.store(row_scale_ptr + block % 4 + padding, 0, (row < scale_m) & (block % 4 + padding < 4))
    column_scale = _inverse_scale(tl.max(tl.abs(values), axis=0))
    column_m: tl.constexpr = tl.cdiv(M, 128) * 128
    tl.store(Column + packed_col[None, :] * column_m + row[:, None], (values / column_scale[None, :]).to(tl.float8e4nv))
    scale_ptr = ColumnScale.to(tl.pointer_type(tl.uint8)) + ((tile // 4) * width + packed_col) * 4
    exponent = (column_scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8)
    tl.store(scale_ptr + tile % 4, exponent)
    if tile == tl.cdiv(M, 128) - 1:
        for padding in tl.static_range(1, 4):
            tl.store(scale_ptr + tile % 4 + padding, 0, tile % 4 + padding < 4)


@triton.jit
def _scale_backward_kernel(
    Partials, QScale, KScale, DQScale, DKScale, Q,
    H: tl.constexpr, TILES: tl.constexpr, SCALE_MIN: tl.constexpr, SCALE_MAX: tl.constexpr,
    MIN_INF: tl.constexpr, MAX_INF: tl.constexpr, BLOCK: tl.constexpr,
):
    scale_min: tl.constexpr = SCALE_MIN if MIN_INF == 0 else (float("-inf") if MIN_INF < 0 else float("inf"))
    scale_max: tl.constexpr = SCALE_MAX if MAX_INF == 0 else (float("-inf") if MAX_INF < 0 else float("inf"))
    head = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    partial = tl.load(Partials + head * TILES + tile, tile < TILES, 0.0)
    grad = tl.sum(partial, axis=0).to(Q.dtype.element_ty).to(tl.float32)
    if head < H:
        scale = tl.load(QScale + head).to(tl.float32)
        tl.store(DQScale + head, tl.where((scale >= scale_min) & (scale <= scale_max), grad, 0.0))
    else:
        head -= H
        scale = tl.load(KScale + head).to(tl.float32)
        tl.store(DKScale + head, tl.where((scale >= scale_min) & (scale <= scale_max), grad, 0.0))


@triton_op("gpt_moe::attention_qkv_norm_rope_backward_quantize", mutates_args=())
def _backward_quantize(
    q: Tensor, k: Tensor, q_scale: Tensor, k_scale: Tensor, cos: Tensor, sin: Tensor,
    q_gain: Tensor, k_gain: Tensor, grad_q: Tensor, grad_k: Tensor, grad_v: Tensor,
    rms: bool, scale_min: float, scale_max: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, heads, seq_len, _ = q.shape
    m, width = batch * seq_len, 3 * heads * 64
    row = torch.empty((m, width), device=q.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((triton.cdiv(width, 512), triton.cdiv(m, 4) * 4), device=q.device, dtype=torch.int32)[:, :m].t()
    column = torch.empty((width, triton.cdiv(m, 128) * 128), device=q.device, dtype=row.dtype)
    column_scale = torch.empty((triton.cdiv(m, 512), width), device=q.device, dtype=torch.int32)
    tiles = triton.cdiv(m, 128)
    partials = torch.empty((2 * heads, tiles), device=q.device, dtype=torch.float32)
    dq_scale = torch.empty_like(q_scale)
    dk_scale = torch.empty_like(k_scale)
    min_inf = (scale_min == float("inf")) - (scale_min == float("-inf"))
    max_inf = (scale_max == float("inf")) - (scale_max == float("-inf"))
    wrap_triton(_backward_quantize_kernel)[(tiles, width // 128)](
        q, k, q_scale, k_scale, cos, sin, q_gain, k_gain, grad_q, grad_k, grad_v,
        row, row_scale, column, column_scale, partials, m, seq_len, heads,
        *q.stride(), *k.stride(), *grad_q.stride(), *grad_k.stride(), *grad_v.stride(),
        *cos.stride()[-2:], *sin.stride()[-2:], rms,
        0.0 if min_inf else scale_min, 0.0 if max_inf else scale_max, min_inf, max_inf,
        num_warps=8, enable_fp_fusion=False,
    )
    wrap_triton(_scale_backward_kernel)[(2 * heads,)](
        partials, q_scale, k_scale, dq_scale, dk_scale, q,
        heads, tiles, 0.0 if min_inf else scale_min, 0.0 if max_inf else scale_max,
        min_inf, max_inf, triton.next_power_of_2(tiles), num_warps=4,
    )
    return row, row_scale, column, column_scale, dq_scale, dk_scale


class _QKVProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_scale, k_scale, cos, sin, eps, rms, scale_min, scale_max, heads,
                weight, weight_scale, transposed, transposed_scale, *weights):
        inputs = x.reshape(-1, x.shape[-1])
        row, row_scale, column, column_scale = _quantize_activations([inputs], True)
        out = _matmul(row, weight, row_scale, weight_scale, torch.bfloat16)
        width = heads * 64
        q, k, v = [out[:, start:start + width].view(x.shape[0], x.shape[1], heads, 64).transpose(1, 2)
                   for start in (0, width, 2 * width)]
        q_out, k_out, q_gain, k_gain = _forward(q, k, q_scale, k_scale, cos, sin, eps, rms, scale_min, scale_max)
        ctx.save_for_backward(column, column_scale, transposed, transposed_scale, q, k,
                              q_scale, k_scale, cos, sin, q_gain, k_gain)
        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weights[0].dtype
        ctx.rms = rms
        ctx.scale_min = scale_min
        ctx.scale_max = scale_max
        ctx.set_materialize_grads(False)
        return q_out, k_out, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        x_column, x_scale, transposed, transposed_scale, q, k, q_scale, k_scale, cos, sin, q_gain, k_gain = ctx.saved_tensors
        gradients = [grad if grad is not None else torch.zeros(q.shape, device=q.device, dtype=q.dtype)
                     for grad in (grad_q, grad_k, grad_v)]
        row, row_scale, column, column_scale, dq_scale, dk_scale = _backward_quantize(
            q, k, q_scale, k_scale, cos, sin, q_gain, k_gain, *gradients,
            ctx.rms, ctx.scale_min, ctx.scale_max,
        )
        grad_x = _matmul(row, transposed, row_scale, transposed_scale, ctx.input_dtype)
        offsets = torch.full((1,), row.shape[0], device=q.device, dtype=torch.int32)
        grad_weight = grouped_fp8_wgrad(column, column_scale, x_column, x_scale, offsets, ctx.weight_dtype)[0]
        width = q.shape[1] * 64
        weight_grads = [grad_weight[start:start + width, :ctx.input_shape[-1]] for start in (0, width, 2 * width)]
        return (grad_x[:, :ctx.input_shape[-1]].reshape(ctx.input_shape), dq_scale, dk_scale,
                None, None, None, None, None, None, None, None, None, None, None, *weight_grads)


def project_qkv_norm_rope(
    x: Tensor, weights: List[Tensor], q_scale: Tensor, k_scale: Tensor, cos: Tensor, sin: Tensor,
    eps: float, rms: bool, scale_min: float, scale_max: float, heads: int,
    prepared_weight: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    if prepared_weight is None:
        with torch.no_grad():
            prepared_weight = quantize_weights(weights)
    return _QKVProjection.apply(x, q_scale, k_scale, cos, sin, eps, rms, scale_min, scale_max, heads,
                                *prepared_weight, *weights)
