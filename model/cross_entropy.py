from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


@triton.jit
def _forward_kernel(
    Logits, Targets, Loss, Statistics,
    V: tl.constexpr, ROW_STRIDE: tl.constexpr, BLOCK: tl.constexpr,
):
    # batch size 8 crosses the signed int32 limit for the flattened logits
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(Targets + row)
    if target == -100:
        tl.store(Loss + row, 0.0)
        tl.store(Statistics + 2 * row, 0.0)
        tl.store(Statistics + 2 * row + 1, 0.0)
    else:
        feature = tl.arange(0, BLOCK)
        maximum = float("-inf")
        denominator = 0.0
        for start in range(0, V, BLOCK):
            column = start + feature
            value = tl.load(Logits + row * ROW_STRIDE + column, column < V, float("-inf")).to(tl.float32)
            next_maximum = tl.maximum(maximum, tl.max(value, 0))
            denominator = denominator * tl.exp(maximum - next_maximum) + tl.sum(tl.exp(value - next_maximum), 0)
            maximum = next_maximum
        log_denominator = tl.log(denominator)
        target_logit = tl.load(Logits + row * ROW_STRIDE + target).to(tl.float32)
        tl.store(Loss + row, (maximum - target_logit) + log_denominator)
        tl.store(Statistics + 2 * row, maximum)
        tl.store(Statistics + 2 * row + 1, log_denominator)


@triton_op("gpt_moe::cross_entropy_forward", mutates_args=())
def _forward(logits: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
    rows, width = logits.shape
    losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
    statistics = torch.empty((rows, 2), device=logits.device, dtype=torch.float32)
    wrap_triton(_forward_kernel)[(rows,)](
        logits, targets, losses, statistics, width, logits.stride(0), 4096, num_warps=8,
    )
    return losses, statistics


@triton.jit
def _backward_kernel(
    Logits, Targets, Statistics, DLoss, DLogits,
    V: tl.constexpr, ROW_STRIDE: tl.constexpr, GRAD_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    target = tl.load(Targets + row)
    if target == -100:
        gradient = tl.full((BLOCK,), 0.0, tl.float32)
    else:
        maximum = tl.load(Statistics + 2 * row)
        log_denominator = tl.load(Statistics + 2 * row + 1)
        value = tl.load(Logits + row * ROW_STRIDE + column, column < V, float("-inf")).to(tl.float32)
        probability = tl.exp((value - maximum) - log_denominator)
        gradient = (probability - (column == target).to(tl.float32)) * tl.load(DLoss + row * GRAD_STRIDE)
    tl.store(DLogits + row * V + column, gradient, column < V)


@triton_op("gpt_moe::cross_entropy_backward", mutates_args=())
def _backward_logits(logits: Tensor, targets: Tensor, statistics: Tensor, grad_loss: Tensor) -> Tensor:
    rows, width = logits.shape
    grad_logits = torch.empty(logits.shape, device=logits.device, dtype=logits.dtype)
    wrap_triton(_backward_kernel)[(rows, triton.cdiv(width, 4096))](
        logits, targets, statistics, grad_loss, grad_logits,
        width, logits.stride(0), grad_loss.stride(0), 4096, num_warps=4,
    )
    return grad_logits


def _setup_context(ctx: Any, inputs: Tuple, output: Tuple[Tensor, Tensor]) -> None:
    logits, targets = inputs
    _, statistics = output
    ctx.save_for_backward(logits, targets, statistics)
    ctx.mark_non_differentiable(statistics)
    ctx.set_materialize_grads(False)


def _backward(ctx: Any, grad_loss: Tensor, _grad_statistics: Optional[Tensor]) -> Tuple:
    logits, targets, statistics = ctx.saved_tensors
    return _backward_logits(logits, targets, statistics, grad_loss), None


_forward.register_autograd(_backward, setup_context=_setup_context)


def cross_entropy(logits: Tensor, targets: Tensor, pad_token_id: Optional[int]) -> Tensor:
    losses, _ = _forward(logits, targets)
    if pad_token_id is not None:
        # match the existing padding mask, including its denominator when targets contain -100
        mask = targets != pad_token_id
        return (losses * mask).sum() / mask.sum()
    return losses.sum() / (targets != -100).sum()
