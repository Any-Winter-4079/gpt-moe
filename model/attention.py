from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask

from .position import Rotary, apply_rotation

if TYPE_CHECKING:
    from config.model import GPTConfig
    from config.training import TrainingConfig
    from .gpt import GPT

# -- Start of (modified) source: https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/mods/softcapping.py --
_PTX_TANH_LOWERED = False

try:
    # use inductor internals if present; otherwise we keep ptx unavailable
    from functools import partial
    from torch._inductor.virtualized import ops
    from torch._inductor.lowering import make_pointwise, register_lowering

    # avoid double-registrations when re-running in dev
    try:
        torch.ops.approx.tanh
        _ALREADY_REGISTERED = True
    except (AttributeError, RuntimeError):
        _ALREADY_REGISTERED = False

    if not _ALREADY_REGISTERED:
        @torch.library.custom_op("approx::tanh", mutates_args=())
        def _tanh_approx(inp: Tensor) -> Tensor:
            # eager fallback path
            return torch.tanh(inp)

        @_tanh_approx.register_fake
        def _(inp: Tensor) -> Tensor:
            return torch.tanh(inp)

        def _tanh_approx_lowering(inp: Tensor) -> Tensor:
            # inline ptx (f32)
            fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
            return make_pointwise(fn)(inp)

        register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)

    _PTX_TANH_LOWERED = True
except Exception:
    # any failure leaves ptx unavailable
    _PTX_TANH_LOWERED = False


class _TanhApproxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        # ptx op is f32; caller upcasts beforehand
        y = torch.ops.approx.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> Tensor:
        (y,) = ctx.saved_tensors
        # d/dx tanh(x) = 1 - tanh^2(x)
        return grad_out * (1 - y * y)


def _tanh_ptx(x: Tensor) -> Tensor:
    # expects float32
    return _TanhApproxFn.apply(x)


def generate_tanh_softcap(soft_cap: float, backend: str):
    """
    returns a score_mod closure for flexattention that applies tanh soft-capping.

    backend: "ptx" | "rational" | "exact"
      - "ptx": use inline ptx tanh.approx (requires inductor lowering to be available)
      - "rational": use polynomial approx (fast & portable)
      - "exact": use torch.tanh
    """
    sc = float(soft_cap)

    if backend == "clamp":
        def fn(score):
            # clamp to [-sc, sc] with no transcendental ops
            return torch.clamp(score, min=-sc, max=sc)
    elif backend == "exact":
        def fn(score):
            return sc * torch.tanh(score / sc)
    elif backend == "rational":
        def fn(score):
            z = (score / sc).to(torch.float32)
            x2 = z * z
            y = z * (27.0 + x2) / (27.0 + 9.0 * x2)
            return (y * sc).to(dtype=score.dtype)
    elif backend == "ptx":
        if not _PTX_TANH_LOWERED:
            raise RuntimeError(
                "tanh_backend='ptx' requested, but ptx lowering is unavailable. "
                "pick 'rational' or 'exact'."
            )

        def fn(score):
            z32 = (score / sc).to(torch.float32)
            y32 = _tanh_ptx(z32)
            return (y32.to(score.dtype) * sc)
    else:
        raise ValueError(f"unknown tanh backend: {backend}")

    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return fn(score)

    tanh_softcap.__name__ = f"tanh_softcap_{backend}_{int(sc)}"
    return tanh_softcap
# -- End of (modified) source: https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/mods/softcapping.py --
################################################
#                QK Norm Debug                 #
################################################


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
        q_eff = torch.clamp(attn.q_scale, max=attn.qk_scale_max).max()
        k_eff = torch.clamp(attn.k_scale, max=attn.qk_scale_max).max()

        max_q_raw = max(max_q_raw, float(q_raw))
        max_k_raw = max(max_k_raw, float(k_raw))
        max_q_eff = max(max_q_eff, float(q_eff))
        max_k_eff = max(max_k_eff, float(k_eff))

    return (
        f"max q_scale raw/eff: {max_q_raw:.4f}/{max_q_eff:.4f} | "
        f"max k_scale raw/eff: {max_k_raw:.4f}/{max_k_eff:.4f} | "
    )

################################################
#                Self Attention                #
################################################


class SelfAttention(nn.Module):
    def __init__(self, gpt_config: GPTConfig, training_config: TrainingConfig, layer_idx: int = 0) -> None:
        super().__init__()

        self.d_model = gpt_config.d_model
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
        self.q_proj = nn.Linear(gpt_config.d_model, self.attn_dim, bias=gpt_config.use_bias)
        self.k_proj = nn.Linear(gpt_config.d_model, self.n_kv_heads * self.head_size, bias=gpt_config.use_bias)
        self.v_proj = nn.Linear(gpt_config.d_model, self.n_kv_heads * self.head_size, bias=gpt_config.use_bias)
        self.c_proj = nn.Linear(self.attn_dim, gpt_config.d_model, bias=gpt_config.use_bias)

        rope_theta = gpt_config.global_rope_theta if is_full_attention else gpt_config.rope_base_theta
        self.rotary = Rotary(self.head_size, rope_theta, training_config.max_seq_len) if gpt_config.pos_encoding_type.lower() == "rope" else None
        
        self.use_qk_norm = gpt_config.use_qk_norm
        self.qk_norm_type = gpt_config.qk_norm_type
        self.qk_scale_max = gpt_config.qk_scale_max
        self.qk_eps = gpt_config.qk_eps

        self.use_bf16_autocast = training_config.use_bf16_autocast
        self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast = training_config.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast

        if self.use_qk_norm:
            # q_scale and k_scale are learned per-head scalars, like LayerNorm’s γ, that let 
            # the model adjust magnitude after normalization
            # They end up controlling the temperature of attention logits
            # How this changes the attention logits:
            # SDPA’s default scaling divides by sqrt(head_size)
            # -> L2 variant (unit-norm Q and K)
            #   Due to ||q'||_2 = ||k'||_2 = 1 (via product with q' and k's gains, i.e, 1/their L2 norm),
            #   (attention) logits = (q' · k') / sqrt(head_size) = 
            #   = (q_scale * k_scale * cosθ) / sqrt(head_size)
            #   i.e., the learned product (q_scale * k_scale) is a per-head temperature multiplier
            #   ---
            #   recalling that q' ⋅ k' = ||q'||_2 ||k'||_2 cosθ
            #   so if you normalize Q and K (gain does that), you get 
            #   ||q'||_2 = ||k'||_2 = 1 if L2 (or = sqrt(head_size) if RMS)
            #   Then the attention logit becomes proportional to cosθ
            #   ---
            # -> RMS variant (unit-RMS Q and K)
            #   because RMS(q) = 1 => L2Norm(q) = sqrt(head_size), we get:
            #   (attention) logits = (q' · k') / sqrt(head_size) =
            #   = (q_scale * sqrt(head_size) * k_scale * sqrt(head_size) * cosθ) / sqrt(head_size) =
            #   = (q_scale * k_scale * sqrt(head_size) * cosθ)
            #   i.e., logits are roughly bounded by sqrt(head_size) * the learned product * [-1, 1]
            #   ---
            #   recalling that RMS(q) = sqrt(1/head_size * sum(q_i^2)) =
            #   = sqrt(1/head_size) * sqrt(sum(q_i^2)) = 
            #   = sqrt(1/head_size) * L2 = 1/sqrt(head_size) * L2
            #   ---
            self.q_scale = nn.Parameter(torch.full((1, self.n_heads, 1, 1), gpt_config.qk_scale_init))
            self.k_scale = nn.Parameter(torch.full((1, self.n_kv_heads, 1, 1), gpt_config.qk_scale_init))
            
        self.is_causal = gpt_config.is_causal
        self.use_flex_attention = gpt_config.use_flex_attention
        self.use_attn_logit_softcapping = gpt_config.use_attn_logit_softcapping
        self.attn_logit_softcap = gpt_config.attn_logit_softcap

        self.tanh_backend = gpt_config.tanh_backend
        self._score_mod = generate_tanh_softcap(self.attn_logit_softcap, backend=self.tanh_backend) if self.use_attn_logit_softcapping else None

    def forward(
            self,
            x: Tensor,
            flex_attn_block_mask: Optional[BlockMask] = None,
            sdpa_attn_mask: Optional[Tensor] = None,
            ) -> Tensor:
        _, seq_len, _ = x.size()

        # cast if no autocast and cast is requested (x input can be f32, and q_proj / k_proj / v_proj weights can be bf16)
        if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and x.dtype != self.q_proj.weight.dtype:
            x = x.to(dtype=self.q_proj.weight.dtype)

        q = self.q_proj(x).view(x.size(0), seq_len, self.n_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(x.size(0), seq_len, self.n_kv_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(x.size(0), seq_len, self.n_kv_heads, self.head_size).transpose(1, 2)

        if self.rotary is not None:
            # apply RoPE
            cos, sin = self.rotary.get_cos_sin(seq_len, q.dtype, q.device)
            q, k = apply_rotation(q, k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            # expand KV to heads if needed
            k = k.unsqueeze(2).expand(x.size(0), self.n_kv_heads, self.q_heads_per_kv_head, seq_len, self.head_size).reshape(x.size(0), self.n_heads, seq_len, self.head_size)
            v = v.unsqueeze(2).expand(x.size(0), self.n_kv_heads, self.q_heads_per_kv_head, seq_len, self.head_size).reshape(x.size(0), self.n_heads, seq_len, self.head_size)

        if self.use_qk_norm:
            if self.qk_norm_type == "rms":
                # 1 / RMS, i.e.,
                # 1 / RMS(q) = 1 / sqrt(mean(q_i^2)) = 1 / (sqrt(1/head_size * sum(q_i^2))) =
                # = sqrt(head_size) / sqrt(mean(q_i^2))
                # multiplying by these will make each query / key vector unit-RMS
                q_gain = torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + self.qk_eps)
                k_gain = torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + self.qk_eps)
            # l2
            else:
                # 1 / L2, i.e.,
                # 1 / L2(q) = 1 / sqrt(sum(q_i^2))
                # multiplying by these will make each query / key vector unit-length
                q_gain = torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + self.qk_eps)
                k_gain = torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + self.qk_eps)

            # clamp scales
            q_scale = torch.clamp(self.q_scale, max=self.qk_scale_max)
            k_scale = torch.clamp(self.k_scale, max=self.qk_scale_max)

            # if we expanded K/V from n_kv_heads -> n_heads, do the same for k_scale
            if self.n_kv_heads != self.n_heads:
                # replicate per-kv scale across the grouped query heads
                k_scale = k_scale.repeat(1, self.q_heads_per_kv_head, 1, 1)

            # keep numerics tidy under autocast
            q_gain = q_gain.to(dtype=q.dtype)
            k_gain = k_gain.to(dtype=k.dtype)
            q_scale = q_scale.to(dtype=q.dtype)
            k_scale = k_scale.to(dtype=k.dtype)

            # (gpu_batch_size, n_heads, seq_len, head_size) * (1, n_heads, 1, 1)
            # gain is the data-dependent inverse norm (L2 or RMS)
            q = q * q_gain * q_scale
            # (gpu_batch_size, n_heads, seq_len, head_size) * (1, n_heads, 1, 1)
            k = k * k_gain * k_scale

        # FlexAttention
        if self.use_flex_attention:
            y = flex_attention(q, k, v, score_mod=self._score_mod, block_mask=flex_attn_block_mask)
        # SDPA
        else:
            is_causal = self.is_causal and (sdpa_attn_mask is None)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_attn_mask, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(x.size(0), seq_len, self.attn_dim)
        # cast if no autocast and cast is requested (y coming from attention can be fp32, proj weights can be bf16)
        if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and y.dtype != self.c_proj.weight.dtype:
            y = y.to(dtype=self.c_proj.weight.dtype)
        y = self.c_proj(y)
        return y
