from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling, Format


class FP8LMHead(te.Linear):
    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__(
            weight.size(1), weight.size(0), bias=False,
            params_dtype=weight.dtype, device=weight.device,
            init_method=lambda _: None,
        )
        # reuse the initialized parameter, including its embedding tie
        self.weight = weight
        # TE 2.20.2 supports current scaling under fullgraph compilation, but not delayed scaling
        self.fp8_recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)

    # current scaling has no persistent scale state; retain the nn.Linear checkpoint format
    get_extra_state = nn.Module.get_extra_state
    set_extra_state = nn.Module.set_extra_state

    def forward(self, hidden: Tensor, use_fp8: bool = False) -> Tensor:
        if not use_fp8:
            return F.linear(hidden, self.weight)
        with te.autocast(enabled=True, recipe=self.fp8_recipe):
            return super().forward(hidden)
