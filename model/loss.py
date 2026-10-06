from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward


@torch.library.custom_op("gpt_moe::fused_linear_cross_entropy", mutates_args=(), device_types="cuda")
def _fused_linear_cross_entropy(
    weight: Tensor,
    hidden: Tensor,
    targets: Tensor,
    ignore_index: int,
    accum_dtype: torch.dtype,
    autocast_dtype: Optional[torch.dtype],
) -> Tuple[Tensor, Tensor, Tensor]:
    # carry autocast explicitly because compiled execution may call this op outside the original context
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
        loss, _, _, _, grad_hidden, grad_weight, _ = fused_linear_cross_entropy_forward(
            _input=hidden,
            weight=weight,
            target=targets,
            ignore_index=ignore_index,
            accum_dtype=accum_dtype,
            compute_gradients=True,
            weight_requires_grad=True,
        )
    return loss, grad_weight, grad_hidden


@_fused_linear_cross_entropy.register_fake
def _fused_linear_cross_entropy_fake(
    weight: Tensor,
    hidden: Tensor,
    targets: Tensor,
    ignore_index: int,
    accum_dtype: torch.dtype,
    autocast_dtype: Optional[torch.dtype],
) -> Tuple[Tensor, Tensor, Tensor]:
    return hidden.new_empty((), dtype=torch.float32), torch.empty_like(weight), torch.empty_like(hidden)


def _setup_context(ctx: Any, inputs: Tuple, output: Tuple[Tensor, Tensor, Tensor]) -> None:
    _, grad_weight, grad_hidden = output
    ctx.save_for_backward(grad_weight, grad_hidden)
    ctx.mark_non_differentiable(grad_weight, grad_hidden)
    # these auxiliary outputs only hold precomputed gradients; avoid allocating gradients for them
    ctx.set_materialize_grads(False)


def _backward(ctx: Any, grad_loss: Tensor, _grad_weight: Optional[Tensor], _grad_hidden: Optional[Tensor]) -> Tuple:
    grad_weight, grad_hidden = ctx.saved_tensors
    # keep upstream loss scaling (including gradient accumulation) in the compiled backward
    return grad_weight * grad_loss, grad_hidden * grad_loss, None, None, None, None


_fused_linear_cross_entropy.register_autograd(_backward, setup_context=_setup_context)


class FusedLinearCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int, accum_dtype: torch.dtype) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.accum_dtype = accum_dtype

    def forward(self, weight: Tensor, hidden: Tensor, targets: Tensor) -> Tensor:
        autocast_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
        loss, _, _ = _fused_linear_cross_entropy(
            weight, hidden, targets, self.ignore_index, self.accum_dtype, autocast_dtype,
        )
        return loss
