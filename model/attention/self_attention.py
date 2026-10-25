from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask

from ..position.rotary import Rotary, apply_rotation
from ..linear.linear import Linear
from ..linear.fp8.custom.dense import project, quantize_weights
from .logit_softcapping import generate_tanh_softcap
from .qkv_norm_rope_fp8 import project_qkv_norm_rope
from .qk_norm_rope import qk_norm_rope

if TYPE_CHECKING:
    from config.model import GPTConfig
    from config.training import TrainingConfig
    from ..gpt import GPT


@torch.no_grad()
def qk_scale_debug_string(model: "GPT") -> str:
    max_q_raw = 0.0
    max_k_raw = 0.0
    max_q_eff = 0.0
    max_k_eff = 0.0

    for blk in model.transformer.h:
        attn = blk.attn
        # if disabled qk_norm, continue
        if not hasattr(attn, "q_scale"):
            continue

        # raw learned params
        q_raw = attn.q_scale.max()
        k_raw = attn.k_scale.max()

        # effective values used in forward (after clamping)
        q_eff = torch.clamp(attn.q_scale, min=attn.qk_scale_min, max=attn.qk_scale_max).max()
        k_eff = torch.clamp(attn.k_scale, min=attn.qk_scale_min, max=attn.qk_scale_max).max()

        max_q_raw = max(max_q_raw, float(q_raw))
        max_k_raw = max(max_k_raw, float(k_raw))
        max_q_eff = max(max_q_eff, float(q_eff))
        max_k_eff = max(max_k_eff, float(k_eff))

    return (
        f"max q_scale raw/eff: {max_q_raw:.4f}/{max_q_eff:.4f} | "
        f"max k_scale raw/eff: {max_k_raw:.4f}/{max_k_eff:.4f} | "
    )


class SelfAttention(nn.Module):
    def __init__(self, gpt_config: GPTConfig, training_config: TrainingConfig, layer_idx: int = 0) -> None:
        super().__init__()

        self.hidden_size = gpt_config.hidden_size
        self.n_heads = gpt_config.n_heads

        # full (global) attention layers use different attn_dim, n_kv_heads, and rope_theta
        is_full_attention = layer_idx in gpt_config.full_attention_layers
        if is_full_attention:
            self.attn_dim = gpt_config.global_attn_dim
            self.n_kv_heads = gpt_config.global_n_kv_heads
        else:
            self.attn_dim = gpt_config.attn_dim
            self.n_kv_heads = gpt_config.n_kv_heads
        self.head_size = self.attn_dim // self.n_heads

        assert 1 <= self.n_kv_heads <= self.n_heads, "n_kv_heads must be in [1, n_heads]"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA/MQA"
        assert self.attn_dim % self.n_heads == 0, "attn_dim must be divisible by n_heads"
        self.q_heads_per_kv_head = self.n_heads // self.n_kv_heads
        self.q_proj = Linear(gpt_config.hidden_size, self.attn_dim, role="q_proj", bias=gpt_config.use_bias)
        self.k_proj = Linear(gpt_config.hidden_size, self.n_kv_heads * self.head_size, role="k_proj", bias=gpt_config.use_bias)
        self.v_proj = Linear(gpt_config.hidden_size, self.n_kv_heads * self.head_size, role="v_proj", bias=gpt_config.use_bias)
        self.out_proj = Linear(self.attn_dim, gpt_config.hidden_size, role="out_proj", bias=gpt_config.use_bias)
        self.prepared_qkv_weight = None
        self.prepared_out_weight = None

        rope_theta = gpt_config.global_rope_theta if is_full_attention else gpt_config.rope_base_theta
        self.rotary = Rotary(self.head_size, rope_theta, training_config.max_seq_len) if gpt_config.pos_encoding_type.lower() == "rope" else None
        
        self.use_qk_norm = gpt_config.use_qk_norm
        self.qk_norm_type = gpt_config.qk_norm_type
        self.qk_scale_min = gpt_config.qk_scale_min
        self.qk_scale_max = gpt_config.qk_scale_max
        self.qk_eps = gpt_config.qk_eps

        self.use_bf16_autocast = training_config.use_bf16_autocast
        self.cast_activations_to_weight_dtype_if_no_autocast = training_config.cast_activations_to_weight_dtype_if_no_autocast

        if self.use_qk_norm:
            # q_scale and k_scale are learned per-head scalars that adjust magnitude after normalization.
            # Let q' and k' denote normalized vectors before applying these scales, and θ their angle.
            # With the default 1 / sqrt(head_size) attention scaling:
            #   logits = q_scale * k_scale * (q' · k') / sqrt(head_size)
            # Ignoring epsilon, the two normalization variants give:
            # - L2: ||q'||_2 = ||k'||_2 = 1
            #   logits = q_scale * k_scale * cosθ / sqrt(head_size)
            # - RMS: ||q'||_2 = ||k'||_2 = sqrt(head_size)
            #   logits = q_scale * k_scale * sqrt(head_size) * cosθ
            self.q_scale = nn.Parameter(torch.full((1, self.n_heads, 1, 1), gpt_config.qk_scale_init))
            self.k_scale = nn.Parameter(torch.full((1, self.n_kv_heads, 1, 1), gpt_config.qk_scale_init))
            
        self.is_causal = gpt_config.is_causal
        self.use_flex_attention = gpt_config.use_flex_attention
        self.use_attn_logit_softcapping = gpt_config.use_attn_logit_softcapping
        self.attn_logit_softcap = gpt_config.attn_logit_softcap

        self.tanh_backend = gpt_config.tanh_backend
        self._score_mod = generate_tanh_softcap(self.attn_logit_softcap, backend=self.tanh_backend) if self.use_attn_logit_softcapping else None

    @torch.no_grad()
    def prepare_fp8_weights(self) -> None:
        self.clear_fp8_weights()
        if all(projection.precision == "fp8" and projection.backend == "custom"
               for projection in (self.q_proj, self.k_proj, self.v_proj)):
            self.prepared_qkv_weight = quantize_weights([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        if self.out_proj.precision == "fp8" and self.out_proj.backend == "custom":
            self.prepared_out_weight = quantize_weights([self.out_proj.weight])

    def clear_fp8_weights(self) -> None:
        self.prepared_qkv_weight = None
        self.prepared_out_weight = None

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        self.clear_fp8_weights()
        super()._load_from_state_dict(*args, **kwargs)

    def forward(
            self,
            x: Tensor,
            flex_attn_block_mask: Optional[BlockMask] = None,
            sdpa_attn_mask: Optional[Tensor] = None,
            ) -> Tensor:
        _, seq_len, _ = x.size()

        # match activation dtype to the receiving module's weight dtype when autocast is disabled
        if not self.use_bf16_autocast and self.cast_activations_to_weight_dtype_if_no_autocast and x.dtype != self.q_proj.weight.dtype:
            x = x.to(dtype=self.q_proj.weight.dtype)

        reuse_weights = self.training and torch.is_grad_enabled()
        fused_qkv = (
            reuse_weights and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
            and self.use_qk_norm and self.rotary is not None
            and self.n_kv_heads == self.n_heads and self.head_size == 64 and self.n_heads % 2 == 0
            and all(projection.precision == "fp8" and projection.backend == "custom" and projection.bias is None
                    for projection in (self.q_proj, self.k_proj, self.v_proj))
        )
        if fused_qkv:
            cos = self.rotary.cos_cached[..., :seq_len, :]
            sin = self.rotary.sin_cached[..., :seq_len, :]
            q, k, v = project_qkv_norm_rope(
                x, [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], self.q_scale, self.k_scale,
                cos, sin, self.qk_eps, self.qk_norm_type == "rms", self.qk_scale_min, self.qk_scale_max,
                self.n_heads, self.prepared_qkv_weight,
            )
        elif all(projection.precision == "fp8" and projection.backend == "custom"
                 for projection in (self.q_proj, self.k_proj, self.v_proj)):
            q, k, v = project(
                x, [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
                self.prepared_qkv_weight if reuse_weights else None,
            )
            if self.q_proj.bias is not None:
                q = q + self.q_proj.bias.to(q.dtype)
                k = k + self.k_proj.bias.to(k.dtype)
                v = v + self.v_proj.bias.to(v.dtype)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError(
                f"attention Q/K/V projections must produce the same dtype; got q={q.dtype}, k={k.dtype}, v={v.dtype}. "
                "Use BF16 autocast or compatible projection precisions/backends."
            )
        if not fused_qkv:
            q = q.view(x.size(0), seq_len, self.n_heads, self.head_size).transpose(1, 2)
            k = k.view(x.size(0), seq_len, self.n_kv_heads, self.head_size).transpose(1, 2)
            v = v.view(x.size(0), seq_len, self.n_kv_heads, self.head_size).transpose(1, 2)

        fused_qk = fused_qkv or (
            self.training and q.is_cuda and q.dtype == torch.bfloat16
            and torch.is_autocast_enabled("cuda")
            and k.dtype == q.dtype and self.rotary is not None and self.use_qk_norm
            and self.n_kv_heads == self.n_heads and self.head_size <= 256
        )
        if fused_qk and not fused_qkv:
            cos = self.rotary.cos_cached[..., :seq_len, :]
            sin = self.rotary.sin_cached[..., :seq_len, :]
            q, k = qk_norm_rope(
                q, k, self.q_scale, self.k_scale, cos, sin, self.qk_eps,
                self.qk_norm_type == "rms", self.qk_scale_min, self.qk_scale_max,
            )
        elif self.rotary is not None and not fused_qkv:
            # apply RoPE
            cos, sin = self.rotary.get_cos_sin(seq_len, q.dtype, q.device)
            q, k = apply_rotation(q, k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            # expand KV to heads if needed
            k = k.unsqueeze(2).expand(x.size(0), self.n_kv_heads, self.q_heads_per_kv_head, seq_len, self.head_size).reshape(x.size(0), self.n_heads, seq_len, self.head_size)
            v = v.unsqueeze(2).expand(x.size(0), self.n_kv_heads, self.q_heads_per_kv_head, seq_len, self.head_size).reshape(x.size(0), self.n_heads, seq_len, self.head_size)

        if self.use_qk_norm and not fused_qk:
            if self.qk_norm_type == "rms":
                # inverse RMS, ignoring epsilon in this derivation:
                # 1 / RMS(q) = 1 / sqrt(mean(q_i^2)) = 1 / (sqrt(1/head_size * sum(q_i^2))) =
                # = sqrt(head_size) / sqrt(sum(q_i^2)).
                # Multiplying by these gains gives approximately unit-RMS vectors when epsilon << mean(q_i^2), and likewise for k
                q_gain = torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + self.qk_eps)
                k_gain = torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + self.qk_eps)
            # l2
            else:
                # inverse L2 norm, ignoring epsilon in this derivation:
                # 1 / L2(q) = 1 / sqrt(sum(q_i^2)).
                # Multiplying by these gains gives approximately unit-length vectors when epsilon << sum(q_i^2), and likewise for k
                q_gain = torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + self.qk_eps)
                k_gain = torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + self.qk_eps)

            # clamp the learned Q/K scales to the configured bounds
            q_scale = torch.clamp(self.q_scale, min=self.qk_scale_min, max=self.qk_scale_max)
            k_scale = torch.clamp(self.k_scale, min=self.qk_scale_min, max=self.qk_scale_max)

            # if we expanded K/V from n_kv_heads -> n_heads, do the same for k_scale
            if self.n_kv_heads != self.n_heads:
                # replicate per-kv scale across the grouped query heads
                k_scale = k_scale.repeat_interleave(self.q_heads_per_kv_head, dim=1)

            # keep numerics tidy under autocast
            q_gain = q_gain.to(dtype=q.dtype)
            k_gain = k_gain.to(dtype=k.dtype)
            q_scale = q_scale.to(dtype=q.dtype)
            k_scale = k_scale.to(dtype=k.dtype)

            # (gpu_batch_size, n_heads, seq_len, head_size) * (gpu_batch_size, n_heads, seq_len, 1) * (1, n_heads, 1, 1)
            # gain is the data-dependent inverse norm (L2 or RMS)
            q = q * q_gain * q_scale
            # (gpu_batch_size, n_heads, seq_len, head_size) * (gpu_batch_size, n_heads, seq_len, 1) * (1, n_heads, 1, 1)
            k = k * k_gain * k_scale

        # FlexAttention
        if self.use_flex_attention:
            y = flex_attention(q, k, v, score_mod=self._score_mod, block_mask=flex_attn_block_mask)
        # SDPA
        else:
            is_causal = self.is_causal and (sdpa_attn_mask is None)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_attn_mask, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(x.size(0), seq_len, self.attn_dim)
        if not self.use_bf16_autocast and self.cast_activations_to_weight_dtype_if_no_autocast and y.dtype != self.out_proj.weight.dtype:
            y = y.to(dtype=self.out_proj.weight.dtype)
        if self.out_proj.precision == "fp8" and self.out_proj.backend == "custom":
            y = project(y, [self.out_proj.weight], self.prepared_out_weight if reuse_weights else None)[0]
            if self.out_proj.bias is not None:
                y = y + self.out_proj.bias.to(y.dtype)
        else:
            y = self.out_proj(y)
        return y
