from __future__ import annotations

import copy
import inspect
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from optimizers.muon import Muon
from optimizers.polar_express import zeropower_via_polar_express


def main():
    assert torch.cuda.is_available(), "run this check in the CUDA training environment"
    source = subprocess.check_output(
        ["git", "show", "f06e05b46c7202bb5f91859b0095ba53aab0292f:optimizers/muon.py"],
        cwd=ROOT, text=True,
    )
    reference = {"__name__": "optimizers.muon_reference", "__package__": "optimizers"}
    exec(compile(source, "<baseline Muon>", "exec"), reference)
    reference["zeropower_backends"]["polarexpress"] = lambda g, steps: zeropower_via_polar_express(g, steps=steps, split_baddbmm=False)
    split_matmul = inspect.signature(zeropower_via_polar_express).parameters["split_baddbmm"].default
    torch.manual_seed(1337)
    shapes = [(32, 32)] * 9 + [(64, 32), (32, 64), (96, 32)]
    for nesterov in (True, False):
        original = [torch.randn(shape, device="cuda") for shape in shapes]
        baseline_params = [torch.nn.Parameter(p.clone()) for p in original]
        candidate_params = [torch.nn.Parameter(p.clone()) for p in original]
        baseline = reference["Muon"](baseline_params, backend="polarexpress", nesterov=nesterov)
        candidate = Muon(candidate_params, backend="polarexpress", nesterov=nesterov)
        for step in range(5):
            if step == 3:
                baseline.load_state_dict(copy.deepcopy(baseline.state_dict()))
                candidate.load_state_dict(copy.deepcopy(candidate.state_dict()))
            for i, (left, right) in enumerate(zip(baseline_params, candidate_params)):
                if step in (0, 2) and i == 0:
                    left.grad = right.grad = None
                else:
                    grad = torch.randn_like(left)
                    left.grad = grad.clone()
                    right.grad = grad.clone()
            before = [p.detach().clone() for p in baseline_params]
            baseline.step()
            candidate.step()
            squared_error = 0.0
            squared_update = 0.0
            for old, left, right in zip(before, baseline_params, candidate_params):
                assert torch.isfinite(right).all()
                squared_error += (left - right).float().square().sum().item()
                squared_update += (left - old).float().square().sum().item()
                if not split_matmul:
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
                if left.grad is not None:
                    torch.testing.assert_close(left.grad, right.grad, rtol=2e-6, atol=2e-7)
                if "momentum_buffer" in baseline.state[left]:
                    torch.testing.assert_close(baseline.state[left]["momentum_buffer"], candidate.state[right]["momentum_buffer"], rtol=2e-6, atol=2e-7)
            relative = (squared_error / max(squared_update, 1e-30)) ** 0.5
            print(f"nesterov={nesterov}, step={step}: weight-difference norm / reference-step norm = {relative:.6g}")
    if split_matmul:
        print("Momentum checks passed. Split-matmul weight differences are reported, not accepted as numerically equivalent; compare training loss before adoption.")
    else:
        print("Muon weight, gradient, momentum and reload checks passed")


if __name__ == "__main__":
    main()
