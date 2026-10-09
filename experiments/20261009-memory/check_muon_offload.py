from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from optimizers.muon import Muon


def main():
    assert torch.cuda.is_available(), "this check requires a CUDA GPU and patch 02"
    torch.manual_seed(1337)
    # include multiple batches, rectangular matrices, and a grouped QKV projection
    shapes = [(32, 32)] * 9 + [(64, 32), (32, 64), (96, 32)]
    for nesterov in (True, False):
        original = [torch.randn(shape, device="cuda") for shape in shapes]
        resident_params = [torch.nn.Parameter(p.clone()) for p in original]
        offloaded_params = [torch.nn.Parameter(p.clone()) for p in original]
        resident = Muon(resident_params, backend="polarexpress", nesterov=nesterov)
        offloaded = Muon(offloaded_params, backend="polarexpress", nesterov=nesterov, offload_momentum=True)
        for step in range(5):
            # optimizer loading may first move CPU momentum back to CUDA
            if step == 3:
                resident.load_state_dict(copy.deepcopy(resident.state_dict()))
                offloaded.load_state_dict(copy.deepcopy(offloaded.state_dict()))
            for i, (left, right) in enumerate(zip(resident_params, offloaded_params)):
                if step in (0, 2) and i == 0:
                    left.grad = right.grad = None
                else:
                    grad = torch.randn_like(left)
                    left.grad = grad.clone()
                    right.grad = grad.clone()
            resident.step()
            offloaded.step()
            torch.cuda.synchronize()
            for left, right in zip(resident_params, offloaded_params):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
                if left.grad is not None:
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                if "momentum_buffer" in resident.state[left]:
                    expected = resident.state[left]["momentum_buffer"]
                    actual = offloaded.state[right]["momentum_buffer"]
                    assert actual.device.type == "cpu" and actual.is_pinned()
                    torch.testing.assert_close(expected.cpu(), actual, rtol=0, atol=0)
        print(f"Muon update and momentum checks passed: nesterov={nesterov}")


if __name__ == "__main__":
    main()
