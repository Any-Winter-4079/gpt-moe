from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..linear.linear import Linear

if TYPE_CHECKING:
    from config.model import GPTConfig
    from config.training import TrainingConfig


class MLP(nn.Module):
    def __init__(self, gpt_config: GPTConfig, training_config: TrainingConfig) -> None:
        super().__init__()
        self.activation_name = gpt_config.activation.lower()
        self.use_bf16_autocast = training_config.use_bf16_autocast
        self.cast_activations_to_weight_dtype_if_no_autocast = training_config.cast_activations_to_weight_dtype_if_no_autocast

        if self.activation_name == "swiglu":
            # without gating:
            # the up projection is (hidden_size, mlp_hidden_dim) with -if used- mlp_hidden_dim biases
            # the down projection is (mlp_hidden_dim, hidden_size) with -if used- hidden_size biases
            # with gating, targeting the same weight parameter count (use_fair_swiglu=True):
            # the combined up projection is (hidden_size, X), and the down projection is (X/2, hidden_size).
            # Gating splits the up-projection output (..., X) into two (..., X/2) tensors
            # and multiplies them elementwise before the down projection.
            # So the 2 * mlp_hidden_dim dimensions without gating need to be split
            # for gating into 3 of size 2 / 3 * mlp_hidden_dim, resulting in:
            # 2 up projection matrices of size (hidden_size, 2 / 3 * mlp_hidden_dim) each, with -if used- 2 * 2 / 3 * mlp_hidden_dim total biases
            # 1 down projection of size (2 / 3 * mlp_hidden_dim, hidden_size) with -if used- hidden_size biases;
            # the total number of weight parameters matches, but, if used, biases -slightly- differ as
            # mlp_hidden_dim + hidden_size != 2 * 2 / 3 * mlp_hidden_dim + hidden_size
            self.hidden_dim = round((2.0/3.0) * gpt_config.mlp_hidden_dim) if gpt_config.use_fair_swiglu else gpt_config.mlp_hidden_dim
            self.up_proj = Linear(gpt_config.hidden_size, self.hidden_dim * 2, role="up_proj", bias=gpt_config.use_bias)
            self.down_proj = Linear(self.hidden_dim, gpt_config.hidden_size, role="down_proj", bias=gpt_config.use_bias)
        else:
            self.hidden_dim = gpt_config.mlp_hidden_dim
            self.up_proj = Linear(gpt_config.hidden_size, self.hidden_dim, role="up_proj", bias=gpt_config.use_bias)
            self.down_proj = Linear(self.hidden_dim, gpt_config.hidden_size, role="down_proj", bias=gpt_config.use_bias)

    def forward(self, x: Tensor) -> Tensor:
        # match activation dtype to the receiving module's weight dtype when autocast is disabled
        if not self.use_bf16_autocast and self.cast_activations_to_weight_dtype_if_no_autocast and x.dtype != self.up_proj.weight.dtype:
            x = x.to(dtype=self.up_proj.weight.dtype)
        x = self.activate(self.up_proj(x))
        # the activation can promote x before the down projection
        if not self.use_bf16_autocast and self.cast_activations_to_weight_dtype_if_no_autocast and x.dtype != self.down_proj.weight.dtype:
            x = x.to(dtype=self.down_proj.weight.dtype)
        return self.down_proj(x)

    def activate(self, x: Tensor) -> Tensor:
        # Gaussian Error Linear Unit
        if self.activation_name == "gelu":
            x = F.gelu(x)
        # Rectified Linear Unit
        elif self.activation_name == "relu":
            x = F.relu(x)
        # Rectified Linear Unit^2
        elif self.activation_name == "relu2":
            x = F.relu(x) ** 2
        # Sigmoid Linear Unit or Swish
        elif self.activation_name == "silu":
            x = F.silu(x)
        # Swish Gated Linear Unit
        elif self.activation_name == "swiglu":
            # split the tensor in two along the last dimension
            x1, x2 = x.chunk(2, dim=-1)
            x = F.silu(x1) * x2
        else:
            raise ValueError(f"unsupported activation function: {self.activation_name}")
        return x
