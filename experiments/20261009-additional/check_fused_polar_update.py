from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from optimizers.muon import Muon
from optimizers.polar_express import fused_aX_plus_BX, polar_express_coeffs, zeropower_via_polar_express


def matrix(shape, strided=False):
    if not strided:
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    batch = (2 * shape[0],) if len(shape) == 3 else ()
    value = torch.randn((*batch, 2 * shape[-1], 2 * shape[-2]), device="cuda", dtype=torch.bfloat16)
    value = value.mT[..., ::2, ::2]
    return value[::2] if len(shape) == 3 else value


def report_difference(name, actual, expected):
    delta = actual.float() - expected.float()
    relative = delta.norm() / expected.float().norm().clamp_min(1e-30)
    print(f"{name}: max_abs={delta.abs().max().item():.6g}, relative_l2={relative.item():.6g}")
    assert torch.isfinite(actual).all()


def main():
    assert torch.cuda.is_available(), "run this check in the CUDA training environment after applying patch 06"
    torch.manual_seed(1337)
    print(f"torch={torch.__version__}, gpu={torch.cuda.get_device_name()}")
    print(f"allow_bf16_reduced_precision_reduction={torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")

    # compare one fused update to FP64 arithmetic on the same BF16 inputs
    for shape in [(1, 1), (17, 29), (65, 131), (131, 65), (1, 33, 67), (3, 65, 131), (8, 128, 256)]:
        for strided in (False, True):
            X = matrix(shape, strided)
            B = matrix((*shape[:-1], shape[-2]), strided)
            B.div_(shape[-2] ** 0.5)
            out = matrix(shape, strided)
            X_before, B_before = X.clone(), B.clone()
            for alpha, _, _ in (polar_express_coeffs[0], polar_express_coeffs[-1]):
                fused_aX_plus_BX(B, X, alpha=alpha, out=out)
                alpha32 = torch.tensor(alpha, dtype=torch.float32).item()
                expected = (B.double() @ X.double() + alpha32 * X.double()).bfloat16()
                torch.testing.assert_close(out, expected, rtol=0.016, atol=0.001)
                report_difference(f"kernel {shape} strided={strided} alpha={alpha:.5g}", out, expected)
            torch.testing.assert_close(X, X_before, rtol=0, atol=0)
            torch.testing.assert_close(B, B_before, rtol=0, atol=0)

    # exercise the compiled five-iteration path, including tall inputs and noncontiguous views
    for shape in [(33, 67), (67, 33), (3, 65, 131), (3, 131, 65)]:
        G = matrix(shape, strided=True)
        before = G.clone()
        baseline = zeropower_via_polar_express(G, fused_update=False).clone()
        fused = zeropower_via_polar_express(G, fused_update=True).clone()
        assert fused.shape == G.shape and fused.dtype == torch.bfloat16
        torch.testing.assert_close(G, before, rtol=0, atol=0)
        report_difference(f"compiled PE {shape}", fused, baseline)
    zeros = torch.zeros((3, 33, 67), device="cuda")
    torch.testing.assert_close(zeropower_via_polar_express(zeros, fused_update=True), zeros.bfloat16(), rtol=0, atol=0)

    # hit the real Muon bucket path, including chunk boundaries, QKV, absent grads and state reload
    shapes = [(32, 32)] * 9 + [(64, 32), (32, 64), (96, 32)]
    for nesterov in (False, True):
        original = [torch.randn(shape, device="cuda") for shape in shapes]
        left = [torch.nn.Parameter(value.clone()) for value in original]
        right = [torch.nn.Parameter(value.clone()) for value in original]
        baseline = Muon(left, backend="polarexpress", nesterov=nesterov, fused_polar_update=False)
        candidate = Muon(right, backend="polarexpress", nesterov=nesterov, fused_polar_update=True)
        for step in range(3):
            if step == 2:
                candidate.load_state_dict(copy.deepcopy(candidate.state_dict()))
                assert candidate.param_groups[0]['fused_polar_update'] is True
            for i, (p, q) in enumerate(zip(left, right)):
                if step == 0 and i == 0:
                    p.grad = q.grad = None
                else:
                    grad = torch.randn_like(p)
                    p.grad, q.grad = grad.clone(), grad.clone()
            before = [p.detach().clone() for p in left]
            baseline.step()
            candidate.step()
            with torch.no_grad():
                drift = torch.cat([(p - q).flatten() for p, q in zip(left, right)]).norm()
                scale = torch.cat([(p - old).flatten() for p, old in zip(left, before)]).norm()
                print(f"Muon nesterov={nesterov} step={step}: weight drift / baseline step norm = {(drift / scale.clamp_min(1e-30)).item():.6g}")
                for p, q in zip(left, right):
                    assert torch.isfinite(q).all()
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                    if 'momentum_buffer' in baseline.state[p]:
                        torch.testing.assert_close(baseline.state[p]['momentum_buffer'], candidate.state[q]['momentum_buffer'], rtol=0, atol=0)
    print("Kernel, compiled execution, gradient, momentum and reload checks passed; PE/weight drift is reported, not accepted as training equivalence.")


if __name__ == "__main__":
    main()
