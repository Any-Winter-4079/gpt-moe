from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..linear.grouped import grouped_linear
from ..linear.grouped_fp8 import grouped_linear_fp8
from ..linear.linear import Linear
from .mlp import MLP

if TYPE_CHECKING:
    from config.model import GPTConfig
    from config.training import TrainingConfig


def _run_experts(experts: nn.ModuleList, parts: Tuple[Tensor, ...]) -> Tensor:
    return torch.cat([expert(part) for expert, part in zip(experts, parts)], dim=0)


class MoE(nn.Module):
    def __init__(self, gpt_config: GPTConfig, training_config: TrainingConfig, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.full_softmax_gating = gpt_config.moe_full_softmax_gating
        self.load_balance_weight = training_config.moe_load_balance_weight
        self.use_bf16_autocast = training_config.use_bf16_autocast
        self.use_grouped_moe = gpt_config.use_grouped_moe
        self.router = Linear(gpt_config.d_model, num_experts, role="router", bias=False)
        self.experts = nn.ModuleList([MLP(gpt_config, training_config) for _ in range(num_experts)])
        self.expert_forward = _run_experts

    def compile_experts(self, **kwargs) -> None:
        self.expert_forward = torch.compile(_run_experts, fullgraph=False, dynamic=True, **kwargs)

    @torch.compiler.disable
    def forward_experts(self, packed: Tensor, counts: Tensor) -> Tensor:
        # isolate variable expert sizes from the surrounding static graph
        parts = packed.split(counts.cpu().tolist())
        return self.expert_forward(self.experts, parts)

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
        # Then y = E_A(x), so dy/dz_A = 0 while A remains selected: token loss cannot train the router this way.
        #
        # full softmax instead gives p_A = exp(z_A) / sum_j exp(z_j).
        # Then y = p_A * E_A(x) and dy/dz_A = p_A * (1 - p_A) * E_A(x).
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
            output = self.forward_packed(hidden_states, top_k_indices, top_k_weights)

        return output.reshape(input_shape), balance_loss

    def forward_packed(self, hidden_states: Tensor, top_k_indices: Tensor, top_k_weights: Tensor) -> Tensor:
        expert_indices, order = top_k_indices.reshape(-1).sort()
        counts = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.int32).scatter_add_(
            0, expert_indices, torch.ones_like(expert_indices, dtype=torch.int32),
        )
        packed = hidden_states[order // self.top_k]
        packed = self.forward_experts(packed, counts)
        inverse_order = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=order.device))
        expert_output = packed[inverse_order].view(hidden_states.shape[0], self.top_k, -1).to(hidden_states.dtype)
        return (expert_output * top_k_weights[:, :, None].to(hidden_states.dtype)).sum(dim=1)

    def forward_grouped(self, hidden_states: Tensor, top_k_indices: Tensor, top_k_weights: Tensor) -> Tensor:
        expert_indices, order = top_k_indices.reshape(-1).sort()
        counts = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.int32).scatter_add_(
            0, expert_indices, torch.ones_like(expert_indices, dtype=torch.int32),
        )
        offsets = counts.cumsum(0, dtype=torch.int32)
        packed = hidden_states[order // self.top_k]

        # stack for computation while keeping each 2d parameter and its Muon update
        up_weight = torch.stack([expert.up_proj.weight for expert in self.experts])
        if self.experts[0].up_proj.precision == "fp8":
            packed = grouped_linear_fp8(packed, up_weight, offsets)
        else:
            dtype = torch.get_autocast_dtype(hidden_states.device.type) if torch.is_autocast_enabled(hidden_states.device.type) else up_weight.dtype
            packed = grouped_linear(packed.to(dtype), up_weight.to(dtype), offsets)
        if self.experts[0].up_proj.bias is not None:
            up_bias = torch.stack([expert.up_proj.bias for expert in self.experts]).to(packed.dtype)
            packed = packed + up_bias[expert_indices]
        packed = self.experts[0].activate(packed)
        down_weight = torch.stack([expert.down_proj.weight for expert in self.experts])
        if self.experts[0].down_proj.precision == "fp8":
            packed = grouped_linear_fp8(packed, down_weight, offsets)
        else:
            dtype = torch.get_autocast_dtype(hidden_states.device.type) if torch.is_autocast_enabled(hidden_states.device.type) else down_weight.dtype
            packed = grouped_linear(packed.to(dtype), down_weight.to(dtype), offsets)
        if self.experts[0].down_proj.bias is not None:
            down_bias = torch.stack([expert.down_proj.bias for expert in self.experts]).to(packed.dtype)
            packed = packed + down_bias[expert_indices]

        # restore token/top-k order before summing, avoiding atomic output accumulation
        inverse_order = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=order.device))
        expert_output = packed[inverse_order].view(hidden_states.shape[0], self.top_k, -1).to(hidden_states.dtype)
        return (expert_output * top_k_weights[:, :, None].to(hidden_states.dtype)).sum(dim=1)
