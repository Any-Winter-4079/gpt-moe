from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.attention.qkv_norm_rope_fp8 import project_qkv_norm_rope
from model.linear.fp8.custom.dense import project, quantize_weights
from model.position.rotary import apply_rotation


def reference(x, weights, q_scale, k_scale, cos, sin, eps, rms, minimum, maximum, heads, prepared):
    q, k, v = [value.view(x.shape[0], x.shape[1], heads, 64).transpose(1, 2)
               for value in project(x, weights, prepared)]
    q, k = apply_rotation(q, k, cos.to(q.dtype), sin.to(k.dtype))
    q_statistic = q.pow(2).mean(-1, keepdim=True) if rms else q.pow(2).sum(-1, keepdim=True)
    k_statistic = k.pow(2).mean(-1, keepdim=True) if rms else k.pow(2).sum(-1, keepdim=True)
    q = q * torch.rsqrt(q_statistic + eps).to(q.dtype) * q_scale.clamp(minimum, maximum).to(q.dtype)
    k = k * torch.rsqrt(k_statistic + eps).to(k.dtype) * k_scale.clamp(minimum, maximum).to(k.dtype)
    return q, k, v


def check(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all(), label
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    print(f"{label}: relative L2={error.item():.8f}")
    assert error < .015, (label, error.item())


torch.manual_seed(2026)
torch.set_float32_matmul_precision("highest")
assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0), "this correctness check requires SM120"

for label, operation in (("eager", project_qkv_norm_rope), ("compiled", torch.compile(project_qkv_norm_rope, fullgraph=True))):
    for dtype in (torch.float32, torch.bfloat16):
        for cached in (False, True):
            for rms in (False, True):
                for batch, length, heads, width in ((1, 17, 2, 128), (2, 129, 4, 129)):
                    torch.compiler.reset()
                    x = torch.randn(batch, length, width * 2, device="cuda", dtype=dtype)[:, :, ::2].requires_grad_()
                    weights = [torch.nn.Parameter(torch.randn(heads * 64, width, device="cuda") * .03) for _ in range(3)]
                    q_scale = torch.nn.Parameter(torch.tensor([-.75, -.5, 1.5, 1.75][:heads], device="cuda").view(1, heads, 1, 1))
                    k_scale = torch.nn.Parameter(torch.tensor([1.75, 1.5, -.5, -.75][:heads], device="cuda").view(1, heads, 1, 1))
                    angle = torch.randn(1, 1, length * 2, 32, device="cuda")
                    cos, sin = angle.cos()[..., ::2, :], angle.sin()[..., ::2, :]
                    prepared = quantize_weights(weights) if cached else None
                    args = (x, weights, q_scale, k_scale, cos, sin, 1e-6, rms, -.5, 1.5, heads, prepared)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        actual = operation(*args)
                        expected = reference(*args)
                    gradients = [torch.randn_like(value) * .01 for value in actual]
                    parameters = [x, *weights, q_scale, k_scale]
                    for output, target in zip(actual, expected):
                        check(output, target, f"{label} cached={cached} rms={rms} output")
                    for selected in ((0, 1, 2), (0,)):
                        actual_grads = torch.autograd.grad([actual[i] for i in selected], parameters,
                                                           [gradients[i] for i in selected], allow_unused=True, retain_graph=True)
                        expected_grads = torch.autograd.grad([expected[i] for i in selected], parameters,
                                                             [gradients[i] for i in selected], allow_unused=True, retain_graph=True)
                        for name, source, gradient, target in zip(("input", "q weight", "k weight", "v weight", "q scale", "k scale"), parameters, actual_grads, expected_grads):
                            gradient = torch.zeros_like(source) if gradient is None else gradient
                            target = torch.zeros_like(source) if target is None else target
                            check(gradient, target, f"{label} cached={cached} rms={rms} {name}")

print("PASS: fused QKV projection, Q/K RoPE and RMS/L2 norm, FP8 backward operands, all parameter gradients, cache reuse and eager/fullgraph compilation")
