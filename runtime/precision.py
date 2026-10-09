from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from model.gpt import GPT


def convert_to_bf16(
        gpt_model: GPT,
        bf16_weights_params_and_scales: Dict[str, Any],
        keep_1d_weights_params_and_scales_in_fp32: bool
        ) -> None:
    # cast all (1d/2d) weights, (learnable) nn.Parameter, (frozen nn.Parameter as) scale
    cast_all = ("all" in bf16_weights_params_and_scales) and (bf16_weights_params_and_scales["all"] is object)
    if cast_all:
        for p in gpt_model.parameters():
            if p.is_floating_point() and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
    # potentially cast some of (1d/2d) weights, (learnable) nn.Parameter, (frozen nn.Parameter as) scale
    else:
        target_classes = tuple(bf16_weights_params_and_scales.values())

        # module classes (e.g., nn.Linear, nn.Embedding)
        module_classes = tuple(cls for cls in target_classes if isinstance(cls, type) and issubclass(cls, nn.Module))
        if module_classes:
            for m in gpt_model.modules():
                if isinstance(m, module_classes):
                    m.bfloat16()

        # nn.Parameter
        if any(cls is nn.Parameter for cls in target_classes):
            for p in gpt_model.parameters():
                if p.is_floating_point() and p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)

    # revert excluded weights, params, scales (as nn.Parameters with requires_grad=False)
    if keep_1d_weights_params_and_scales_in_fp32:
        for p in gpt_model.parameters():
            if p.dim() == 1 and p.dtype == torch.bfloat16:
                p.data = p.data.float()


def configure_adamw_moments(optimizer: torch.optim.AdamW, use_bf16_adamw_moments: bool) -> None:
    dtype = torch.bfloat16 if use_bf16_adamw_moments else torch.float32
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.dtype != dtype:
                if not (p.dtype == torch.float32 and dtype == torch.bfloat16
                        and p.device.type == "cuda" and group["fused"]):
                    raise ValueError("mixed-dtype AdamW requires fused CUDA with fp32 parameters and bf16 moments")
                if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 13):
                    raise ValueError("bf16 AdamW moments with fp32 parameters require PyTorch 2.13 or newer")

            state = optimizer.state[p]
            if not state:
                # initialize directly in the requested dtype, before the first optimizer step
                state["step"] = torch.zeros(
                    (), dtype=torch.float32,
                    device=p.device if group["fused"] or group["capturable"] else "cpu",
                )
                state["exp_avg"] = torch.zeros_like(p, dtype=dtype)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=dtype)
            else:
                # keep the current moment values and step counter when restoring state
                state["exp_avg"] = state["exp_avg"].to(dtype=dtype)
                state["exp_avg_sq"] = state["exp_avg_sq"].to(dtype=dtype)


def precision_context(training_config, device_type: str):
    if training_config.use_all_bf16_and_null_ctx or not training_config.use_bf16_autocast:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
