from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor


def _quantize_fp8(x: Tensor, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    # dynamic tensor scaling, as in torchao/float8/float8_scaling_utils.py
    # the minimum only guards zero/tiny tensors; it is not a calibrated model scale
    scale = torch.finfo(dtype).max / x.float().abs().amax().clamp_min(1e-12)
    return (x.float() * scale).to(dtype).contiguous(), scale.reciprocal()


@torch.compile(fullgraph=True)
def _forward_impl(hidden: Tensor, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    hidden_fp8, hidden_inv_scale = _quantize_fp8(hidden, torch.float8_e4m3fn)
    weight_fp8, weight_inv_scale = _quantize_fp8(weight, torch.float8_e4m3fn)
    logits = torch._scaled_mm(
        hidden_fp8, weight_fp8.t(),
        scale_a=hidden_inv_scale, scale_b=weight_inv_scale,
        out_dtype=torch.bfloat16, use_fast_accum=False,
    )
    return logits, hidden_fp8, weight_fp8, hidden_inv_scale, weight_inv_scale


@torch.library.custom_op("gpt_moe::fp8_lm_head", mutates_args=(), device_types="cuda")
def _fp8_mm(hidden: Tensor, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if torch.cuda.get_device_capability(hidden.device) < (8, 9):
        raise ValueError("use_fp8_lm_head requires an NVIDIA GPU with compute capability 8.9 or newer")
    with torch.autocast("cuda", enabled=False):
        return _forward_impl(hidden, weight)


@_fp8_mm.register_fake
def _fp8_mm_fake(hidden: Tensor, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        hidden.new_empty((hidden.size(0), weight.size(0)), dtype=torch.bfloat16),
        hidden.new_empty(hidden.shape, dtype=torch.float8_e4m3fn),
        weight.new_empty(weight.shape, dtype=torch.float8_e4m3fn),
        hidden.new_empty((), dtype=torch.float32),
        weight.new_empty((), dtype=torch.float32),
    )


@torch.compile(fullgraph=True)
def _backward_impl(
        grad: Tensor,
        hidden_fp8: Tensor,
        weight_fp8: Tensor,
        hidden_inv_scale: Tensor,
        weight_inv_scale: Tensor,
        hidden_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        ) -> Tuple[Tensor, Tensor]:
    # measure the incoming gradient after CE and accumulation scaling
    grad_fp8, grad_inv_scale = _quantize_fp8(grad, torch.float8_e5m2)
    grad_hidden = torch._scaled_mm(
        grad_fp8, weight_fp8.t().contiguous().t(),
        scale_a=grad_inv_scale, scale_b=weight_inv_scale,
        out_dtype=hidden_dtype, use_fast_accum=False,
    )
    # use modded-nanogpt's wide weight-gradient GEMM ordering
    # https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-01-13_Fp8LmHead
    grad_weight = torch._scaled_mm(
        hidden_fp8.t().contiguous(), grad_fp8.t().contiguous().t(),
        scale_a=hidden_inv_scale, scale_b=grad_inv_scale,
        out_dtype=torch.float32, use_fast_accum=False,
    ).t().to(weight_dtype)
    return grad_hidden, grad_weight


@torch.library.custom_op("gpt_moe::fp8_lm_head_backward", mutates_args=(), device_types="cuda")
def _fp8_mm_backward(
        grad: Tensor,
        hidden_fp8: Tensor,
        weight_fp8: Tensor,
        hidden_inv_scale: Tensor,
        weight_inv_scale: Tensor,
        hidden_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        ) -> Tuple[Tensor, Tensor]:
    with torch.autocast("cuda", enabled=False):
        return _backward_impl(
            grad, hidden_fp8, weight_fp8, hidden_inv_scale, weight_inv_scale, hidden_dtype, weight_dtype,
        )


@_fp8_mm_backward.register_fake
def _fp8_mm_backward_fake(
        grad: Tensor,
        hidden_fp8: Tensor,
        weight_fp8: Tensor,
        hidden_inv_scale: Tensor,
        weight_inv_scale: Tensor,
        hidden_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        ) -> Tuple[Tensor, Tensor]:
    return (
        hidden_fp8.new_empty(hidden_fp8.shape, dtype=hidden_dtype),
        weight_fp8.new_empty((weight_fp8.size(1), weight_fp8.size(0)), dtype=weight_dtype).t(),
    )


def _setup_context(ctx: Any, inputs: Tuple, output: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> None:
    hidden, weight = inputs
    _, hidden_fp8, weight_fp8, hidden_inv_scale, weight_inv_scale = output
    ctx.save_for_backward(hidden_fp8, weight_fp8, hidden_inv_scale, weight_inv_scale)
    ctx.dtypes = hidden.dtype, weight.dtype
    ctx.mark_non_differentiable(hidden_fp8, weight_fp8, hidden_inv_scale, weight_inv_scale)
    ctx.set_materialize_grads(False)


def _backward(ctx: Any, grad_logits: Tensor, *_: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    return _fp8_mm_backward(grad_logits, *ctx.saved_tensors, *ctx.dtypes)


_fp8_mm.register_autograd(_backward, setup_context=_setup_context)


def fp8_lm_head(
        hidden: Tensor,
        weight: Tensor,
        ) -> Tensor:
    flattened = hidden.reshape(-1, hidden.size(-1))
    num_tokens, hidden_size = flattened.shape
    vocab_size = weight.shape[0]
    if any(size % 16 for size in (num_tokens, hidden_size, vocab_size)):
        raise ValueError("use_fp8_lm_head requires token count, hidden size and vocabulary size divisible by 16")
    logits, _, _, _, _ = _fp8_mm(flattened, weight)
    return logits.view(*hidden.shape[:-1], vocab_size)
