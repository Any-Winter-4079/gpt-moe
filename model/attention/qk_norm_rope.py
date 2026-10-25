from __future__ import annotations

from typing import Any, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


@triton.jit
def _rotate(x, paired, cos, sin, feature, dtype: tl.constexpr, inverse: tl.constexpr):
    # retain the existing product rounding before adding each RoPE pair
    first = (x * cos).to(dtype).to(tl.float32)
    second = (paired * sin).to(dtype).to(tl.float32)
    subtract = (feature % 2 == 1) if inverse else (feature % 2 == 0)
    return (first + tl.where(subtract, -second, second)).to(dtype).to(tl.float32)


@triton.jit
def _forward_kernel(
    Q, K, QScale, KScale, Cos, Sin, QOut, KOut, QGain, KGain,
    T: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
    qb: tl.constexpr, qh: tl.constexpr, qt: tl.constexpr, qd: tl.constexpr,
    kb: tl.constexpr, kh: tl.constexpr, kt: tl.constexpr, kd: tl.constexpr,
    ct: tl.constexpr, cd: tl.constexpr, st: tl.constexpr, sd: tl.constexpr,
    EPS: tl.constexpr, RMS: tl.constexpr,
    SCALE_MIN: tl.constexpr, SCALE_MAX: tl.constexpr,
    MIN_INF: tl.constexpr, MAX_INF: tl.constexpr,
    ROWS: tl.constexpr, BLOCK_D: tl.constexpr,
):
    scale_min: tl.constexpr = SCALE_MIN if MIN_INF == 0 else (float("-inf") if MIN_INF < 0 else float("inf"))
    scale_max: tl.constexpr = SCALE_MAX if MAX_INF == 0 else (float("-inf") if MAX_INF < 0 else float("inf"))
    batch_head = tl.program_id(0)
    batch = batch_head // (2 * H)
    head = batch_head % (2 * H)
    token = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    feature = tl.arange(0, BLOCK_D)
    valid = (token[:, None] < T) & (feature[None, :] < D)
    if head < H:
        source = Q + batch * qb + head * qh + token[:, None] * qt
        x = tl.load(source + feature[None, :] * qd, valid, 0.0).to(tl.float32)
        scale = tl.load(QScale + head).to(tl.float32)
        out = QOut + ((batch * H + head) * T + token[:, None]) * D + feature[None, :]
        gain_out = QGain + (batch * H + head) * T + token
    else:
        head -= H
        source = K + batch * kb + head * kh + token[:, None] * kt
        x = tl.load(source + feature[None, :] * kd, valid, 0.0).to(tl.float32)
        scale = tl.load(KScale + head).to(tl.float32)
        out = KOut + ((batch * H + head) * T + token[:, None]) * D + feature[None, :]
        gain_out = KGain + (batch * H + head) * T + token
    dtype: tl.constexpr = Q.dtype.element_ty
    pair_index = tl.broadcast_to((feature[None, :] ^ 1), (ROWS, BLOCK_D))
    paired = tl.gather(x, pair_index, axis=1)
    cos = tl.load(Cos + token[:, None] * ct + (feature[None, :] // 2) * cd, valid, 0.0).to(dtype).to(tl.float32)
    sin = tl.load(Sin + token[:, None] * st + (feature[None, :] // 2) * sd, valid, 0.0).to(dtype).to(tl.float32)
    rotated = _rotate(x, paired, cos, sin, feature[None, :], dtype, False)
    statistic = tl.sum(rotated * rotated, axis=1)
    if RMS:
        statistic /= D
    gain = tl.rsqrt(statistic + EPS)
    scale = tl.where(scale != scale, scale, tl.minimum(tl.maximum(scale, scale_min), scale_max)).to(dtype).to(tl.float32)
    normalized = (rotated * gain.to(dtype).to(tl.float32)[:, None]).to(dtype).to(tl.float32)
    tl.store(out, normalized * scale, valid)
    tl.store(gain_out, gain, token < T)


@triton.jit
def _backward_kernel(
    Q, K, QScale, KScale, Cos, Sin, QGain, KGain, DQOut, DKOut, DQ, DK, DScalePart,
    T: tl.constexpr, D: tl.constexpr, H: tl.constexpr, TILES: tl.constexpr,
    qb: tl.constexpr, qh: tl.constexpr, qt: tl.constexpr, qd: tl.constexpr,
    kb: tl.constexpr, kh: tl.constexpr, kt: tl.constexpr, kd: tl.constexpr,
    dqb: tl.constexpr, dqh: tl.constexpr, dqt: tl.constexpr, dqd: tl.constexpr,
    dkb: tl.constexpr, dkh: tl.constexpr, dkt: tl.constexpr, dkd: tl.constexpr,
    ct: tl.constexpr, cd: tl.constexpr, st: tl.constexpr, sd: tl.constexpr,
    RMS: tl.constexpr, SCALE_MIN: tl.constexpr, SCALE_MAX: tl.constexpr,
    MIN_INF: tl.constexpr, MAX_INF: tl.constexpr,
    ROWS: tl.constexpr, BLOCK_D: tl.constexpr,
):
    scale_min: tl.constexpr = SCALE_MIN if MIN_INF == 0 else (float("-inf") if MIN_INF < 0 else float("inf"))
    scale_max: tl.constexpr = SCALE_MAX if MAX_INF == 0 else (float("-inf") if MAX_INF < 0 else float("inf"))
    batch_head = tl.program_id(0)
    batch = batch_head // (2 * H)
    head = batch_head % (2 * H)
    token = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    feature = tl.arange(0, BLOCK_D)
    valid = (token[:, None] < T) & (feature[None, :] < D)
    if head < H:
        source = Q + batch * qb + head * qh + token[:, None] * qt
        x = tl.load(source + feature[None, :] * qd, valid, 0.0).to(tl.float32)
        scale = tl.load(QScale + head).to(tl.float32)
        gain = tl.load(QGain + (batch * H + head) * T + token, token < T, 0.0).to(tl.float32)
        grad = tl.load(DQOut + batch * dqb + head * dqh + token[:, None] * dqt + feature[None, :] * dqd, valid, 0.0).to(tl.float32)
        out = DQ + ((batch * H + head) * T + token[:, None]) * D + feature[None, :]
    else:
        head -= H
        source = K + batch * kb + head * kh + token[:, None] * kt
        x = tl.load(source + feature[None, :] * kd, valid, 0.0).to(tl.float32)
        scale = tl.load(KScale + head).to(tl.float32)
        gain = tl.load(KGain + (batch * H + head) * T + token, token < T, 0.0).to(tl.float32)
        grad = tl.load(DKOut + batch * dkb + head * dkh + token[:, None] * dkt + feature[None, :] * dkd, valid, 0.0).to(tl.float32)
        out = DK + ((batch * H + head) * T + token[:, None]) * D + feature[None, :]
    dtype: tl.constexpr = Q.dtype.element_ty
    pair_index = tl.broadcast_to((feature[None, :] ^ 1), (ROWS, BLOCK_D))
    paired = tl.gather(x, pair_index, axis=1)
    cos = tl.load(Cos + token[:, None] * ct + (feature[None, :] // 2) * cd, valid, 0.0).to(dtype).to(tl.float32)
    sin = tl.load(Sin + token[:, None] * st + (feature[None, :] // 2) * sd, valid, 0.0).to(dtype).to(tl.float32)
    rotated = _rotate(x, paired, cos, sin, feature[None, :], dtype, False)
    rounded_gain = gain.to(dtype).to(tl.float32)
    normalized = (rotated * rounded_gain[:, None]).to(dtype).to(tl.float32)
    scale_part = (grad * normalized).to(dtype).to(tl.float32)
    tl.store(DScalePart + batch_head * TILES + tl.program_id(1), tl.sum(tl.sum(scale_part, axis=1), axis=0))
    scale = tl.where(scale != scale, scale, tl.minimum(tl.maximum(scale, scale_min), scale_max)).to(dtype).to(tl.float32)
    grad = (grad * scale).to(dtype).to(tl.float32)
    grad_gain = tl.sum((grad * rotated).to(dtype).to(tl.float32), axis=1).to(dtype).to(tl.float32)
    # autocast keeps the norm branch in FP32 until its contribution returns to Q/K
    grad_statistic = (grad_gain * -0.5) * (gain * gain * gain)
    if RMS:
        grad_statistic /= D
    direct = (grad * rounded_gain[:, None]).to(dtype).to(tl.float32)
    indirect = (grad_statistic[:, None] * (2.0 * rotated)).to(dtype).to(tl.float32)
    grad_rotated = (direct + indirect).to(dtype).to(tl.float32)
    paired_grad = tl.gather(grad_rotated, pair_index, axis=1)
    grad_x = _rotate(grad_rotated, paired_grad, cos, sin, feature[None, :], dtype, True)
    tl.store(out, grad_x, valid)


@triton.jit
def _scale_backward_kernel(
    Partials, QScale, KScale, DQScale, DKScale, Q,
    B: tl.constexpr, H: tl.constexpr, TILES: tl.constexpr,
    SCALE_MIN: tl.constexpr, SCALE_MAX: tl.constexpr,
    MIN_INF: tl.constexpr, MAX_INF: tl.constexpr, BLOCK: tl.constexpr,
):
    scale_min: tl.constexpr = SCALE_MIN if MIN_INF == 0 else (float("-inf") if MIN_INF < 0 else float("inf"))
    scale_max: tl.constexpr = SCALE_MAX if MAX_INF == 0 else (float("-inf") if MAX_INF < 0 else float("inf"))
    head = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    partial = tl.load(Partials + (index // TILES) * (2 * H * TILES) + head * TILES + index % TILES, index < B * TILES, 0.0)
    grad = tl.sum(partial, axis=0).to(Q.dtype.element_ty).to(tl.float32)
    if head < H:
        raw_scale = tl.load(QScale + head).to(tl.float32)
        tl.store(DQScale + head, tl.where((raw_scale >= scale_min) & (raw_scale <= scale_max), grad, 0.0))
    else:
        head -= H
        raw_scale = tl.load(KScale + head).to(tl.float32)
        tl.store(DKScale + head, tl.where((raw_scale >= scale_min) & (raw_scale <= scale_max), grad, 0.0))


@triton_op("gpt_moe::qk_norm_rope_forward", mutates_args=())
def _forward(
    q: Tensor, k: Tensor, q_scale: Tensor, k_scale: Tensor, cos: Tensor, sin: Tensor,
    eps: float, rms: bool, scale_min: float, scale_max: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, heads, seq_len, head_size = q.shape
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    q_gain = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
    k_gain = torch.empty_like(q_gain)
    rows = 16 if head_size <= 64 else 8
    # keep infinities out of Inductor's generated Python metadata
    min_inf = (scale_min == float("inf")) - (scale_min == float("-inf"))
    max_inf = (scale_max == float("inf")) - (scale_max == float("-inf"))
    wrap_triton(_forward_kernel)[(batch * 2 * heads, triton.cdiv(seq_len, rows))](
        q, k, q_scale, k_scale, cos, sin, q_out, k_out, q_gain, k_gain,
        seq_len, head_size, heads, *q.stride(), *k.stride(), *cos.stride()[-2:], *sin.stride()[-2:],
        eps, rms, 0.0 if min_inf else scale_min, 0.0 if max_inf else scale_max,
        min_inf, max_inf, rows, triton.next_power_of_2(head_size),
        num_warps=4, enable_fp_fusion=False,
    )
    return q_out, k_out, q_gain, k_gain


@triton_op("gpt_moe::qk_norm_rope_backward", mutates_args=())
def _backward_kernel_call(
    q: Tensor, k: Tensor, q_scale: Tensor, k_scale: Tensor, cos: Tensor, sin: Tensor,
    q_gain: Tensor, k_gain: Tensor, grad_q: Tensor, grad_k: Tensor,
    rms: bool, scale_min: float, scale_max: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, heads, seq_len, head_size = q.shape
    rows = 16 if head_size <= 64 else 8
    tiles = triton.cdiv(seq_len, rows)
    dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    dq_scale = torch.empty_like(q_scale)
    dk_scale = torch.empty_like(k_scale)
    partials = torch.empty((batch, 2 * heads, tiles), device=q.device, dtype=torch.float32)
    min_inf = (scale_min == float("inf")) - (scale_min == float("-inf"))
    max_inf = (scale_max == float("inf")) - (scale_max == float("-inf"))
    wrap_triton(_backward_kernel)[(batch * 2 * heads, tiles)](
        q, k, q_scale, k_scale, cos, sin, q_gain, k_gain, grad_q, grad_k, dq, dk, partials,
        seq_len, head_size, heads, tiles, *q.stride(), *k.stride(), *grad_q.stride(), *grad_k.stride(),
        *cos.stride()[-2:], *sin.stride()[-2:], rms,
        0.0 if min_inf else scale_min, 0.0 if max_inf else scale_max, min_inf, max_inf,
        rows, triton.next_power_of_2(head_size), num_warps=4, enable_fp_fusion=False,
    )
    wrap_triton(_scale_backward_kernel)[(2 * heads,)](
        partials, q_scale, k_scale, dq_scale, dk_scale, q,
        batch, heads, tiles, 0.0 if min_inf else scale_min, 0.0 if max_inf else scale_max,
        min_inf, max_inf, triton.next_power_of_2(batch * tiles), num_warps=4,
    )
    return dq, dk, dq_scale, dk_scale


def _setup_context(ctx: Any, inputs: Tuple, output: Tuple) -> None:
    q, k, q_scale, k_scale, cos, sin, _, ctx.rms, ctx.scale_min, ctx.scale_max = inputs
    ctx.mark_non_differentiable(*output[2:])
    ctx.save_for_backward(q, k, q_scale, k_scale, cos, sin, *output[2:])


def _backward(ctx: Any, grad_q: Tensor, grad_k: Tensor, *unused: Tensor) -> Tuple:
    dq, dk, dq_scale, dk_scale = _backward_kernel_call(
        *ctx.saved_tensors, grad_q, grad_k, ctx.rms, ctx.scale_min, ctx.scale_max,
    )
    return dq, dk, dq_scale, dk_scale, None, None, None, None, None, None


_forward.register_autograd(_backward, setup_context=_setup_context)


def qk_norm_rope(
    q: Tensor, k: Tensor, q_scale: Tensor, k_scale: Tensor, cos: Tensor, sin: Tensor,
    eps: float, rms: bool, scale_min: float, scale_max: float,
) -> Tuple[Tensor, Tensor]:
    return _forward(q, k, q_scale, k_scale, cos, sin, eps, rms, scale_min, scale_max)[:2]
