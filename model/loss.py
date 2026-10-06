from __future__ import annotations

import torch
from torch import Tensor
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


class FusedLinearCrossEntropyLoss(LigerFusedLinearCrossEntropyLoss):
    # keep Liger's scalar reads and custom backward outside the compiled transformer
    @torch.compiler.disable
    def forward(self, weight: Tensor, hidden: Tensor, targets: Tensor) -> Tensor:
        return super().forward(weight, hidden, targets)
