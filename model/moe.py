from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .grouped_linear import grouped_linear
from .mlp import MLP

if TYPE_CHECKING:
    from config.model import GPTConfig
    from config.training import TrainingConfig


class MoE(nn.Module):
    def __init__(self, gpt_config: GPTConfig, training_config: TrainingConfig, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.full_softmax_gating = gpt_config.moe_full_softmax_gating
        self.load_balance_weight = training_config.moe_load_balance_weight
        self.use_bf16_autocast = training_config.use_bf16_autocast
        self.use_grouped_moe = gpt_config.use_grouped_moe
        self.router = nn.Linear(gpt_config.d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(gpt_config, training_config) for _ in range(num_experts)])

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        input_shape = x.shape
        # flatten batch size x sequence length
        hidden_states = x.reshape(-1, x.shape[-1])

        router_input = hidden_states
        if not self.use_bf16_autocast and router_input.dtype != self.router.weight.dtype:
            router_input = router_input.to(dtype=self.router.weight.dtype)
        # (batch size x sequence length, num_experts)
        router_logits = self.router(router_input).float()
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_probs = None
        # z_i is expert i's router logit, E_i(x) its output, and y the MoE output.
        #
        # if only A is selected, softmax over A gives q_A = exp(z_A) / exp(z_A) = 1.
        # then y = E_A(x), so dy/dz_A = 0 while A remains selected: token loss cannot train the router this way.
        #
        # full softmax instead gives p_A = exp(z_A) / sum_j exp(z_j).
        # then y = p_A * E_A(x) and dy/dz_A = p_A * (1 - p_A) * E_A(x).
        #
        # with A and B selected, selected-only softmax also has a gradient:
        # dy/dz_A = q_A * (1 - q_A) * (E_A(x) - E_B(x)).
        if self.top_k == 1 or self.full_softmax_gating:
            router_probs = F.softmax(router_logits, dim=-1)
            top_k_weights = router_probs.gather(-1, top_k_indices)
        else:
            top_k_weights = F.softmax(top_k_logits, dim=-1)

        balance_loss = None
        if self.training and self.load_balance_weight > 0:
            if router_probs is None:
                router_probs = F.softmax(router_logits, dim=-1)
            mean_probs = router_probs.mean(dim=0)
            expert_counts = torch.zeros_like(mean_probs).scatter_add_(
                0, top_k_indices.reshape(-1),
                torch.ones_like(top_k_indices, dtype=mean_probs.dtype).reshape(-1),
            )
            expert_fraction = expert_counts / top_k_indices.numel()
            balance_loss = self.num_experts * (expert_fraction * mean_probs).sum()

        if self.use_grouped_moe:
            output = self.forward_grouped(hidden_states, top_k_indices, top_k_weights)
        else:
            output = torch.zeros_like(hidden_states)
            for expert_idx, expert in enumerate(self.experts):
                token_idx, top_k_pos = torch.where(top_k_indices == expert_idx)
                expert_output = expert(hidden_states[token_idx])
                weighted_output = expert_output.to(output.dtype) * top_k_weights[token_idx, top_k_pos, None].to(output.dtype)
                output.index_add_(0, token_idx, weighted_output)

        return output.reshape(input_shape), balance_loss

    def forward_grouped(self, hidden_states: Tensor, top_k_indices: Tensor, top_k_weights: Tensor) -> Tensor:
        expert_indices, order = top_k_indices.reshape(-1).sort()
        counts = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.int32).scatter_add_(
            0, expert_indices, torch.ones_like(expert_indices, dtype=torch.int32),
        )
        offsets = counts.cumsum(0, dtype=torch.int32)
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else self.experts[0].c_fc.weight.dtype
        packed = hidden_states[order // self.top_k].to(dtype)

        # stack for computation while keeping each 2d parameter and its Muon update
        up_weight = torch.stack([expert.c_fc.weight for expert in self.experts]).to(dtype)
        packed = grouped_linear(packed, up_weight, offsets)
        if self.experts[0].c_fc.bias is not None:
            up_bias = torch.stack([expert.c_fc.bias for expert in self.experts]).to(dtype)
            packed = packed + up_bias[expert_indices]
        packed = self.experts[0].activate(packed)
        down_weight = torch.stack([expert.c_proj.weight for expert in self.experts]).to(dtype)
        packed = grouped_linear(packed, down_weight, offsets)
        if self.experts[0].c_proj.bias is not None:
            down_bias = torch.stack([expert.c_proj.bias for expert in self.experts]).to(dtype)
            packed = packed + down_bias[expert_indices]

        # restore token/top-k order before summing, avoiding atomic output accumulation
        inverse_order = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=order.device))
        expert_output = packed[inverse_order].view(hidden_states.shape[0], self.top_k, -1).to(hidden_states.dtype)
        return (expert_output * top_k_weights[:, :, None].to(hidden_states.dtype)).sum(dim=1)
