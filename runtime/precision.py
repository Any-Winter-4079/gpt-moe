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


def precision_context(training_config, device_type: str):
    if training_config.use_all_bf16_and_null_ctx or not training_config.use_bf16_autocast:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
