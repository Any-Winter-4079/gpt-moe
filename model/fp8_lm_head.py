from __future__ import annotations

from typing import Tuple

import torch
import triton
from torch import Tensor
from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.fused_linear_cross_entropy import MAX_FUSED_SIZE


@torch.compile(fullgraph=True)
def _quantize_fp8(x: Tensor, scale: float, dtype: torch.dtype) -> Tensor:
    return (x.float() * scale).to(dtype)


def fp8_linear_cross_entropy_forward(
        weight: Tensor,
        hidden: Tensor,
        targets: Tensor,
        ignore_index: int,
        ) -> Tuple[Tensor, Tensor, Tensor]:
    if torch.cuda.get_device_capability(hidden.device) < (8, 9):
        raise ValueError("use_fp8_lm_head requires an NVIDIA GPU with compute capability 8.9 or newer")
    num_tokens, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    if any(size % 16 for size in (num_tokens, hidden_size, vocab_size)):
        raise ValueError("use_fp8_lm_head requires token count, hidden size and vocabulary size divisible by 16")

    # modded-nanogpt's FP8 head uses E4M3 operands with fixed hidden/weight scales 2 and 32
    # https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-01-13_Fp8LmHead
    weight_fp8 = _quantize_fp8(weight, 32.0, torch.float8_e4m3fn)
    weight_fp8_col = weight_fp8.t().contiguous().t()
    hidden_inv_scale = hidden.new_tensor(0.5, dtype=torch.float32)
    weight_inv_scale = hidden.new_tensor(1.0 / 32, dtype=torch.float32)

    # quantize unreduced CE gradients so their scale does not depend on batch size or accumulation
    mean_scale = (targets != ignore_index).sum(dtype=torch.float32).clamp_min_(1).reciprocal_()
    grad_inv_scale = mean_scale / 32768
    grad_hidden = torch.empty_like(hidden)
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)
    losses = torch.zeros(num_tokens, device=hidden.device, dtype=torch.float32)

    # keep Liger's token chunking instead of retaining the full tokens-by-vocabulary matrix
    inc_factor = triton.cdiv(vocab_size, hidden_size)
    chunk_size = max(16, triton.next_power_of_2(triton.cdiv(num_tokens, inc_factor)))
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        hidden_fp8 = _quantize_fp8(hidden[start:end], 2.0, torch.float8_e4m3fn)
        logits = torch._scaled_mm(
            hidden_fp8, weight_fp8.t(),
            scale_a=hidden_inv_scale, scale_b=weight_inv_scale,
            out_dtype=torch.bfloat16, use_fast_accum=True,
        )
        # Liger computes CE in fp32 and replaces the bf16 logits with their gradients
        liger_cross_entropy_kernel[(end - start,)](
            X_ptr=logits, X_stride=logits.stride(0),
            Y_ptr=targets[start:end], Y_stride=targets.stride(0),
            weight_ptr=None, loss_ptr=losses[start:end], z_loss_ptr=None, loss_stride=1,
            token_accuracy_ptr=None, token_accuracy_stride=0,
            predicted_tokens_ptr=None, predicted_tokens_stride=0,
            n_cols=vocab_size, n_non_ignore=1, sum_non_ignore_weight=1, weight_sum=0.0,
            ignore_index=ignore_index, lse_square_scale=0.0, label_smoothing=0.0,
            reduction="sum", softcap=None,
            RETURN_Z_LOSS=False, RETURN_TOKEN_ACCURACY=False, RETURN_PREDICTED_TOKENS=False,
            HAS_WEIGHT=False, HAS_SOFTCAPPING=False, HAS_GRADIENTS=True,
            BLOCK_SIZE=min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size)), num_warps=32,
        )
        grad_fp8 = _quantize_fp8(logits, 32768.0, torch.float8_e5m2)
        del logits
        grad_hidden[start:end] = torch._scaled_mm(
            grad_fp8, weight_fp8_col,
            scale_a=grad_inv_scale, scale_b=weight_inv_scale,
            out_dtype=torch.bfloat16, use_fast_accum=False,
        )
        # upstream's transposed order makes the head weight-gradient GEMM wide rather than tall
        grad_weight.add_(torch._scaled_mm(
            hidden_fp8.t().contiguous(), grad_fp8.t().contiguous().t(),
            scale_a=hidden_inv_scale, scale_b=grad_inv_scale,
            out_dtype=torch.float32, use_fast_accum=False,
        ).t())
        del hidden_fp8, grad_fp8

    return losses.sum() * mean_scale, grad_weight.to(weight.dtype), grad_hidden
