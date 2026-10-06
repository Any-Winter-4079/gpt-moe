from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import sys
import time
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import triton
from torch import Tensor
from torch.nn import functional as F

from config.model import GPTConfig


# fixed settings for the failing run, without importing the tokenizer or training entry point
SEED = 1337
COMPILE_MODE = "max-autotune-no-cudagraphs"
GRAD_ACCUM_MINI_STEPS = 4
SHAPES = {"4x8192": (4, 8192), "8x8192": (8, 8192), "4x16384": (4, 16384)}
GRAD_REL_L2_TOLERANCE = 0.02


class HeadLoss(nn.Module):
    def __init__(self, weight: Tensor, return_logits: bool) -> None:
        super().__init__()
        self.lm_head = nn.Linear(weight.size(1), weight.size(0), bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(weight)
        self.return_logits = return_logits

    def forward(self, hidden: Tensor, targets: Tensor) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        # match the dense training head, fp32 cross-entropy, and both observed return conventions
        logits = self.lm_head(hidden)
        logits_for_loss = logits.float() if logits.dtype != torch.float32 else logits
        loss = F.cross_entropy(logits_for_loss.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        token_loss = loss
        balance_term = loss.new_zeros(())
        return (logits, loss, token_loss, balance_term) if self.return_logits else (loss, token_loss, balance_term)


def run_case(
        weight: Tensor,
        hidden: Tensor,
        targets: Tensor,
        batch_size: int,
        seq_len: int,
        compiled: bool,
        return_logits: bool,
        ) -> Dict[str, Tensor]:
    label = f"{'compiled' if compiled else 'eager'} | return_logits: {return_logits}"
    print(f"starting {label}", flush=True)
    model = HeadLoss(weight, return_logits).cuda()
    num_tokens = batch_size * seq_len
    x = hidden[:num_tokens].view(batch_size, seq_len, -1).cuda().requires_grad_(True)
    y = targets[:num_tokens].view(batch_size, seq_len).cuda()
    forward = torch.compile(model, mode=COMPILE_MODE, fullgraph=True, dynamic=False) if compiled else model

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_t = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = forward(x, y)
    loss = output[1] if return_logits else output[0]
    # retain returned logits through backward, as in the current training loop
    (loss / GRAD_ACCUM_MINI_STEPS).backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_t
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    # retain only the small loss and gradients on CPU between cases
    result = {
        "loss": loss.detach().cpu(),
        "hidden.grad": x.grad.detach().cpu(),
        "lm_head.weight.grad": model.lm_head.weight.grad.detach().cpu(),
    }
    print(
        f"finished {label} | loss: {result['loss'].item():.8f} | "
        f"forward/backward time including initial compilation: {elapsed:.2f} s | peak allocated: {peak_gib:.2f} GiB",
        flush=True,
    )
    return result


def compare_results(actual: Dict[str, Tensor], reference: Dict[str, Tensor]) -> bool:
    matched = True
    for name, expected in reference.items():
        observed = actual[name]
        nan_count = torch.isnan(observed).sum().item()
        inf_count = torch.isinf(observed).sum().item()
        if nan_count or inf_count:
            print(f"  {name}: NON-FINITE | nan count: {nan_count} | inf count: {inf_count}", flush=True)
            matched = False
            continue

        diff = observed - expected
        max_abs_error = diff.abs().max().item()
        expected_norm = torch.linalg.vector_norm(expected.reshape(-1), dtype=torch.float64).item()
        observed_norm = torch.linalg.vector_norm(observed.reshape(-1), dtype=torch.float64).item()
        diff_norm = torch.linalg.vector_norm(diff.reshape(-1), dtype=torch.float64).item()
        denominator = max(expected_norm, torch.finfo(torch.float64).tiny)
        rel_l2_error = diff_norm / denominator
        norm_ratio = observed_norm / denominator
        if name == "loss":
            close = torch.isclose(observed, expected, rtol=1e-5, atol=1e-6).item()
        else:
            # a relative norm test catches zeroed gradients even when individual values are tiny
            close = rel_l2_error <= GRAD_REL_L2_TOLERANCE
        print(
            f"  {name}: {'MATCH' if close else 'MISMATCH'} | max abs error: {max_abs_error:.6e} | "
            f"relative L2 error: {rel_l2_error:.6e} | compiled/eager norm: {norm_ratio:.6e}",
            flush=True,
        )
        matched = matched and close
    return matched


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare eager and compiled dense output-head/cross-entropy gradients on one GPU")
    parser.add_argument("--shapes", nargs="+", choices=list(SHAPES), default=list(SHAPES))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU")

    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    gpt_config = GPTConfig()
    print(f"python: {sys.version}", flush=True)
    print(f"torch: {torch.__version__} | CUDA build: {torch.version.cuda} | triton: {triton.__version__}", flush=True)
    print(f"gpu: {torch.cuda.get_device_name(0)} | capability: {torch.cuda.get_device_capability(0)}", flush=True)
    print(
        f"seed: {SEED} | d_model: {gpt_config.d_model} | vocab: {gpt_config.vocab_size} | "
        f"fp32 weights/inputs | bf16 autocast | fp32 cross-entropy | loss divided by {GRAD_ACCUM_MINI_STEPS}",
        flush=True,
    )
    print(f"compile mode: {COMPILE_MODE} | fullgraph: True | dynamic: False | float32 matmul precision: high", flush=True)
    print(f"gradient MATCH threshold: relative L2 error <= {GRAD_REL_L2_TOLERANCE}; loss rtol: 1e-5, atol: 1e-6", flush=True)
    print("synthetic head-only diagnostic; no attention, DDP, optimizer updates, or training-speed measurement", flush=True)

    # use one flat input pool so both 65,536-token shapes contain exactly the same values
    generator = torch.Generator().manual_seed(SEED)
    weight = torch.randn(gpt_config.vocab_size, gpt_config.d_model, generator=generator) * 0.02
    hidden = torch.randn(65536, gpt_config.d_model, generator=generator)
    targets = torch.randint(0, gpt_config.pad_token_id, (65536,), generator=generator)
    all_matched = True
    for shape in args.shapes:
        batch_size, seq_len = SHAPES[shape]
        num_elements = batch_size * seq_len * gpt_config.vocab_size
        print(f"\nshape: {shape} | logits elements: {num_elements:,} | exceeds signed int32: {num_elements > 2**31 - 1}", flush=True)
        reference = run_case(weight, hidden, targets, batch_size, seq_len, compiled=False, return_logits=False)
        torch.cuda.empty_cache()
        for name, value in reference.items():
            if not torch.isfinite(value).all().item():
                raise FloatingPointError(f"eager reference has non-finite {name} for shape {shape}")
        for return_logits in (False, True):
            actual = run_case(weight, hidden, targets, batch_size, seq_len, compiled=True, return_logits=return_logits)
            torch.cuda.empty_cache()
            matched = compare_results(actual, reference)
            all_matched = all_matched and matched
            del actual
        del reference

    print(
        "\nAll comparisons matched within the stated tolerances; this does not rule out a full-model compilation issue."
        if all_matched else "\nAt least one compiled result differed from eager execution; see the per-tensor results above.",
        flush=True,
    )
    return 0 if all_matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
