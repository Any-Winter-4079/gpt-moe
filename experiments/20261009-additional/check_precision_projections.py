from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.model import GPTConfig


def tensor_scale(x, dynamic, fixed, maximum):
    shape = (x.shape[0],) if x.ndim == 3 else ()
    if not dynamic:
        return torch.full(shape, fixed, device=x.device, dtype=torch.float32)
    if not x.numel():
        return torch.ones(shape, device=x.device)
    return x.float().abs().amax(dim=(-2, -1)).clamp_min(1e-12) / maximum


def quantized_rows(x, scale, precision):
    x = x.float()
    if precision == "fp8":
        return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    rows, width = x.shape
    padded = F.pad(x, (0, (-width) % 16)).reshape(rows, -1, 16) if rows else x.new_empty((0, (width + 15) // 16, 16))
    if padded.numel() == 0:
        return x
    block_scale = (padded.abs().amax(-1) / (6 * scale)).clamp(2 ** -9, 448).to(torch.float8_e4m3fn).float()
    normalized = padded / (scale * block_scale[..., None])
    magnitude = normalized.abs()
    # nearest-even rounding to the E2M1 levels, including subnormals
    rounded = torch.where(magnitude <= 2, (magnitude * 2).round() / 2,
                          torch.where(magnitude <= 4, magnitude.round(), (magnitude / 2).round() * 2))
    values = rounded.clamp_max(6) * normalized.sign() * scale * block_scale[..., None]
    return values.reshape(rows, -1)[:, :width]


def reference(x, weight, grad, counts, precision, dynamic, backward):
    maximum = 448 if precision == "fp8" else 6 * 448
    sx = tensor_scale(x, dynamic, 0.01, maximum)
    sw = tensor_scale(weight, dynamic, 0.01, maximum)
    sg = tensor_scale(grad, dynamic, 0.01, maximum)
    outputs, gx, gw = [], [], []
    start = 0
    for expert, count in enumerate(counts):
        a, w, g = x[start:start + count], weight[expert], grad[start:start + count]
        weight_scale = sw[expert]
        outputs.append(quantized_rows(a, sx, precision) @ quantized_rows(w, weight_scale, precision).T)
        if backward:
            gx.append(quantized_rows(g, sg, precision) @ quantized_rows(w.T, weight_scale, precision).T)
            gw.append(quantized_rows(g.T, sg, precision) @ quantized_rows(a.T, sx, precision).T)
        else:
            gx.append(g.float() @ w.float())
            gw.append(g.float().T @ a.float())
        start += count
    return torch.cat(outputs).to(x.dtype), torch.cat(gx).to(x.dtype), torch.stack(gw).to(weight.dtype)


def compare(actual, expected):
    for a, e in zip(actual, expected):
        assert a.shape == e.shape and a.dtype == e.dtype
        assert torch.isfinite(a).all()
        if e.numel():
            max_error = (a.float() - e.float()).abs().max()
            tolerance = 0.04 * e.float().abs().max().clamp_min(1e-30)
            assert max_error <= tolerance, (max_error.item(), tolerance.item())


def main():
    parser = argparse.ArgumentParser(description="Optional CUDA validation; apply one precision patch first")
    parser.add_argument("--precision", choices=("fp8", "nvfp4"), required=True)
    args = parser.parse_args()
    module = importlib.import_module(f"model.{args.precision}_linear")
    config = GPTConfig()
    setattr(config, f"{args.precision}_projections", "both")
    getattr(module, f"validate_{args.precision}_config")(config)
    op = getattr(module, f"grouped_{args.precision}_linear")
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [((0, 1, 15, 17), 0.1, True), ((0, 1, 15, 17), 0.1, False),
             ((0, 0, 0, 0), 0.1, True), ((0, 1, 0, 17), 1e-9, True),
             ((0, 1, 0, 17), 0.0, True)]
    for case, (counts, amplitude, dynamic) in enumerate(cases):
        # non-contiguous source tensors exercise both projection orientations
        x = (torch.randn(sum(counts), 160, device="cuda", dtype=torch.bfloat16) * amplitude)[:, ::2].detach().requires_grad_()
        w = (torch.randn(len(counts), 96, 160, device="cuda", dtype=torch.bfloat16) * amplitude)[:, :, ::2].detach().requires_grad_()
        grad = torch.randn(sum(counts), 96, device="cuda", dtype=torch.bfloat16) * 0.1
        offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        for backward in (False, True):
            options = (backward, dynamic, 1.0, 0.01, 0.01, 0.01)
            def eager(a, b):
                return op(a, b, offsets, *options)
            def checkpointed(a, b):
                return checkpoint(eager, a, b, use_reentrant=False)
            expected = reference(x.detach(), w.detach(), grad, counts, args.precision, dynamic, backward)
            for name, fn in (("eager", eager), ("compiled", torch.compile(eager, fullgraph=True)),
                             ("compiled checkpoint", torch.compile(checkpointed, fullgraph=True))):
                result = fn(x, w)
                gx, gw = torch.autograd.grad(result, (x, w), grad_outputs=grad)
                compare((result, gx, gw), expected)
                assert torch.count_nonzero(gw[0]) == 0
                print(f"case={case} backward={backward} {name}: passed")
            torch._dynamo.reset()
        if case == 0:
            # inspect the same forward kernel specialization for native instructions
            sx = module._tensor_scale(x.detach(), dynamic, 1.0, 0.01)
            sw = module._tensor_scale(w.detach(), dynamic, 1.0, 0.01)
            b = w.detach().transpose(-2, -1)
            output = x.new_empty((x.shape[0], w.shape[1]))
            workers = torch.cuda.get_device_properties(x.device).multi_processor_count
            compiled = getattr(module, f"_{args.precision}_mm_kernel")[(workers,)](
                x.detach(), b, output, offsets, sx, sw, x.shape[0], w.shape[1], x.shape[1],
                x.stride(0), x.stride(1), b.stride(0), b.stride(-2), b.stride(-1),
                len(counts), workers, False, BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
                num_warps=4, num_stages=1,
            )
            ptx = compiled.asm["ptx"]
            assert "mma.sync" in ptx
            if args.precision == "fp8":
                assert ".e4m3.e4m3" in ptx
            else:
                assert "kind::mxf4nvf4" in ptx and ".e2m1.e2m1" in ptx and ".ue4m3" in ptx
            print("native instruction check: passed")
    torch.cuda.synchronize()
    print("projection checks passed; complete model, optimizer resume and training quality still require separate validation")


if __name__ == "__main__":
    main()
