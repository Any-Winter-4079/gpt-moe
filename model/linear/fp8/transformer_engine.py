from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling, Format


class TransformerEngineLinear(te.Linear):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__(
            in_features, out_features, bias=bias,
            params_dtype=torch.get_default_dtype(), device="cpu",
            init_method=lambda _: None,
        )
        if not bias:
            self.bias = None
        # preserve nn.Linear initialization and its random-number sequence
        nn.Linear.reset_parameters(self)
        self.fp8_recipe = Float8BlockScaling(
            fp8_format=Format.E4M3,
            x_block_scaling_dim=1,
            w_block_scaling_dim=2,
            grad_block_scaling_dim=1,
        )

    # block scaling has no persistent scale state; retain ordinary weight/bias checkpoint keys
    get_extra_state = nn.Module.get_extra_state
    set_extra_state = nn.Module.set_extra_state

    def forward_fp8(self, x: Tensor) -> Tensor:
        with te.autocast(enabled=True, recipe=self.fp8_recipe):
            return super().forward(x)
