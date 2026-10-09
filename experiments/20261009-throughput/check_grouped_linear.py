from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.grouped_linear import grouped_linear


def main():
    assert torch.cuda.is_available(), "run this check in the CUDA training environment"
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("highest")
    compiled = torch.compile(grouped_linear, fullgraph=True)
    for dtype in (torch.float32, torch.bfloat16):
        tolerance = dict(rtol=5e-4, atol=5e-4) if dtype == torch.float32 else dict(rtol=0.03, atol=0.05)
        for counts, inputs, outputs in (([0, 17, 65, 3], 64, 96), ([130, 0, 0, 5], 96, 64)):
            offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
            for transpose_pack in (False, True):
                for fn in (grouped_linear, compiled):
                    # exercise strided inputs, empty experts and both weight layouts
                    x = torch.randn(sum(counts), inputs * 2, device="cuda", dtype=dtype)[:, ::2].detach().requires_grad_()
                    weight = torch.randn(len(counts), outputs, inputs, device="cuda", dtype=dtype, requires_grad=True)
                    reference_x = x.detach().clone().requires_grad_()
                    reference_weight = weight.detach().clone().requires_grad_()
                    packed_weight = torch.stack([w.t() for w in weight]).transpose(-2, -1) if transpose_pack else weight
                    actual = fn(x, packed_weight, offsets)
                    chunks = []
                    start = 0
                    for expert, count in enumerate(counts):
                        chunks.append(F.linear(reference_x[start:start + count], reference_weight[expert]))
                        start += count
                    expected = torch.cat(chunks)
                    grad = torch.randn(outputs, sum(counts), device="cuda", dtype=dtype).t()
                    actual_grads = torch.autograd.grad(actual, (x, weight), grad)
                    expected_grads = torch.autograd.grad(expected, (reference_x, reference_weight), grad)
                    for left, right in zip((actual, *actual_grads), (expected, *expected_grads)):
                        assert torch.isfinite(left).all()
                        torch.testing.assert_close(left, right, **tolerance)
                        print(f"dtype={dtype}, transpose_pack={transpose_pack}, max_abs_error={(left.float() - right.float()).abs().max().item():.6g}")
                    for expert, count in enumerate(counts):
                        if count == 0:
                            assert torch.count_nonzero(actual_grads[1][expert]).item() == 0
    print("Grouped forward/backward checks passed, eager and compiled")


if __name__ == "__main__":
    main()
