"""Tensor-only CUDA checks after applying 08-fused-lion.patch; no model or training run."""
import argparse
import copy
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    import torch

    spec = importlib.util.spec_from_file_location('fused_lion_experiment', args.root / 'optimizers/lion.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(123)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        p = torch.nn.Parameter(torch.randn(1031, device='cuda', dtype=dtype))
        absent = torch.nn.Parameter(torch.ones(3, device='cuda', dtype=dtype))
        initial_absent = absent.detach().clone()
        reference = p.detach().clone()
        momentum = torch.zeros_like(p, dtype=torch.float32)
        optimizer = module.FusedLion([p, absent], lr=0.03, betas=(0.8, 0.97), weight_decay=0.2)
        for index in range(7):
            grad = torch.randn_like(p)
            if index == 0:
                grad[:9] = 0
            p.grad = grad
            update = momentum * 0.8 + grad.float() * 0.2
            reference.copy_((reference.float() * (1 - 0.03 * 0.2) - 0.03 * update.sign()).to(dtype))
            momentum.mul_(0.97).add_(grad.float(), alpha=0.03)
            optimizer.step()
            tolerance = 1e-6 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
            torch.testing.assert_close(p, reference, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(optimizer.state[p]['exp_avg'], momentum, atol=1e-7, rtol=1e-6)
        assert absent not in optimizer.state
        torch.testing.assert_close(absent, initial_absent, atol=0, rtol=0)

        saved = copy.deepcopy(optimizer.state_dict())
        resumed_p = torch.nn.Parameter(p.detach().clone())
        resumed_absent = torch.nn.Parameter(absent.detach().clone())
        resumed = module.FusedLion([resumed_p, resumed_absent])
        resumed.load_state_dict(saved)
        assert resumed.state[resumed_p]['exp_avg'].dtype == torch.float32
        torch.testing.assert_close(resumed.state[resumed_p]['exp_avg'], optimizer.state[p]['exp_avg'], atol=0, rtol=0)
        for _ in range(3):
            grad = torch.randn_like(p)
            p.grad, resumed_p.grad = grad, grad.clone()
            optimizer.step()
            resumed.step()
            torch.testing.assert_close(resumed_p, p, atol=0, rtol=0)
            torch.testing.assert_close(resumed.state[resumed_p]['exp_avg'], optimizer.state[p]['exp_avg'], atol=0, rtol=0)

        # warmup restores an empty optimizer state and must recreate momentum
        cold = module.FusedLion([p])
        empty = copy.deepcopy(cold.state_dict())
        cold.step()
        cold.load_state_dict(empty)
        assert not cold.state
        print(f'Lion {dtype}: FP32-equation parity, zero gradient, missing gradient, reload and warmup reset passed')


if __name__ == '__main__':
    main()
