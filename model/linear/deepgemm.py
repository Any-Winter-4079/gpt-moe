from __future__ import annotations

import torch
from torch import Tensor
from torch.library import custom_op


@custom_op("gpt_moe::deepgemm_fp8_wgrad", mutates_args=())
def grouped_fp8_wgrad(
    grad: Tensor, grad_scale: Tensor, x: Tensor, x_scale: Tensor,
    offsets: Tensor, dtype: torch.dtype,
) -> Tensor:
    # DeepGEMM b64107f with docker_build/cu134/deepgemm-sm120-psum.patch
    import deep_gemm

    if torch.cuda.get_device_capability(grad.device) != (12, 0):
        raise ValueError("DeepGEMM grouped FP8 weight gradients require SM120; select the triton backend on other GPUs")
    if grad.shape[0] % 4 or x.shape[0] % 4:
        raise ValueError("DeepGEMM grouped FP8 projection widths must be multiples of 4")
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("DeepGEMM grouped FP8 weight gradients require float32 or bfloat16 weights")
    out = torch.empty((offsets.numel(), grad.shape[0], x.shape[0]), device=grad.device, dtype=dtype)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (grad, grad_scale), (x, x_scale), out, None, offsets,
        recipe=(1, 1, 128), use_psum_layout=True,
    )
    return out


@grouped_fp8_wgrad.register_fake
def _fake_grouped_fp8_wgrad(
    grad: Tensor, grad_scale: Tensor, x: Tensor, x_scale: Tensor,
    offsets: Tensor, dtype: torch.dtype,
) -> Tensor:
    return torch.empty((offsets.numel(), grad.shape[0], x.shape[0]), device=grad.device, dtype=dtype)
