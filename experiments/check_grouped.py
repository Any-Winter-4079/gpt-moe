from pathlib import Path
import argparse
import inspect
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.linear.grouped import grouped_linear
from model.linear import grouped_fp8

grouped_mlp_fp8_relu2 = None
if Path(grouped_fp8.__file__).with_name("grouped_fp8_mlp.py").is_file():
    from model.linear.grouped_fp8_mlp import grouped_mlp_fp8_relu2


def quantized_reference(x, block_rows):
    m, k = x.shape
    padded = F.pad(x.detach().float(), (0, (-k) % 128, 0, (-m) % block_rows))
    values = padded.reshape(padded.shape[0] // block_rows, block_rows, padded.shape[1] // 128, 128)
    amax = values.abs().amax((1, 3), keepdim=True)
    scale = torch.where(amax == 0, 1.0, torch.exp2(torch.ceil(torch.log2(amax / 448.0))).clamp_min(2.0**-126))
    return ((values / scale).to(torch.float8_e4m3fn).float() * scale).reshape(padded.shape)[:m, :k]


def fp8_reference(x, weight, grad, counts):
    outputs, input_grads, weight_grads = [], [], []
    start = 0
    for expert, count in enumerate(counts):
        group_x = x[start:start + count]
        group_grad = grad[start:start + count]
        quantized_weight = quantized_reference(weight[expert], 128)
        outputs.append((quantized_reference(group_x, 1) @ quantized_weight.t()).bfloat16())
        input_grads.append(quantized_reference(group_grad, 1) @ quantized_weight)
        weight_grads.append(quantized_reference(group_grad.t(), 1) @ quantized_reference(group_x.t(), 1).t())
        start += count
    return torch.cat(outputs), torch.cat(input_grads), torch.stack(weight_grads)


def run(operation, x, weight, offsets, grad):
    x = x.detach().clone().requires_grad_()
    weight = weight.detach().clone().requires_grad_()
    output = operation(x, weight, offsets)
    output.backward(grad)
    assert output.dtype == torch.bfloat16
    assert x.grad.dtype == x.dtype and weight.grad.dtype == weight.dtype
    return output.detach(), x.grad.detach(), weight.grad.detach()


def check(actual, expected, label, tolerance):
    for weight_grad in actual[2:]:
        for expert, count in enumerate(counts):
            if count == 0:
                assert not weight_grad[expert].count_nonzero(), f"{label}: empty expert {expert} received a weight gradient"
    names = ("output", "input gradient", "up weight gradient", "down weight gradient") if len(actual) == 4 else ("output", "input gradient", "weight gradient")
    for name, value, reference in zip(names, actual, expected):
        assert value.shape == reference.shape and torch.isfinite(value).all(), (label, name)
        difference = (value.float() - reference.float()).norm()
        relative_error = difference / reference.float().norm().clamp_min(1e-12)
        print(f"{label} {name}: relative L2={relative_error.item():.8f}")
        assert relative_error < tolerance, (label, name, relative_error.item())


def bf16_operation(x, weight, offsets):
    return grouped_linear(x.bfloat16(), weight.bfloat16(), offsets)


def unfused_mlp(x, up_weight, down_weight, offsets, wgrad_backend):
    hidden = grouped_fp8.grouped_linear_fp8(x, up_weight, offsets, wgrad_backend)
    return grouped_fp8.grouped_linear_fp8(F.relu(hidden) ** 2, down_weight, offsets, wgrad_backend)


def fused_mlp(x, up_weight, down_weight, offsets, wgrad_backend):
    return grouped_mlp_fp8_relu2(x, up_weight, down_weight, offsets, wgrad_backend)


def run_mlp(operation, x, up_weight, down_weight, offsets, grad, wgrad_backend):
    inputs = [value.detach().clone().requires_grad_() for value in (x, up_weight, down_weight)]
    output = operation(*inputs, offsets, wgrad_backend)
    grads = torch.autograd.grad(output, inputs, grad)
    assert output.dtype == torch.bfloat16
    assert all(value.dtype == source.dtype for value, source in zip(grads, inputs))
    return (output.detach(), *(value.detach() for value in grads))


if "projection_backend" in inspect.signature(grouped_fp8.grouped_linear_fp8).parameters:
    def fp8_operation(x, weight, offsets):
        return grouped_fp8.grouped_linear_fp8(x, weight, offsets, "deepgemm", "deepgemm")

    def fp8_triton_wgrad_operation(x, weight, offsets):
        return grouped_fp8.grouped_linear_fp8(x, weight, offsets, "triton", "deepgemm")
else:
    def fp8_operation(x, weight, offsets):
        return grouped_fp8.grouped_linear_fp8(x, weight, offsets, "deepgemm")

    def fp8_triton_wgrad_operation(x, weight, offsets):
        return grouped_fp8.grouped_linear_fp8(x, weight, offsets, "triton")


parser = argparse.ArgumentParser()
parser.add_argument("--triton-wgrad", action="store_true")
args = parser.parse_args()
wgrad_backends = ("deepgemm", "triton") if args.triton_wgrad else ("deepgemm",)

torch.manual_seed(2026)
torch.set_float32_matmul_precision("highest")
assert torch.cuda.is_available(), "this correctness check requires CUDA"
assert torch.cuda.get_device_capability() == (12, 0), "these experiments require SM120"
counts = [0, 1, 0, 128, 134, 0, 257, 0]
offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)

for width, output_width in ((128, 256), (256, 128)):
    x = torch.randn(width, sum(counts), device="cuda").t()
    x[:2] = 0
    weight = torch.randn(len(counts), output_width, width, device="cuda") * .03
    weight[3, :128, :128] = 0
    grad = torch.randn(sum(counts), output_width, device="cuda", dtype=torch.bfloat16) * .01
    assert not x.is_contiguous()

    def bf16_reference(x, weight, offsets):
        pieces, start = [], 0
        for expert, count in enumerate(counts):
            pieces.append(F.linear(x[start:start + count].bfloat16(), weight[expert].bfloat16()))
            start += count
        return torch.cat(pieces)

    expected_bf16 = run(bf16_reference, x, weight, offsets, grad)
    expected_fp8 = fp8_reference(x, weight, grad, counts)
    for label, operation in (("eager", lambda fn: fn), ("compiled", lambda fn: torch.compile(fn, fullgraph=True))):
        suffix = f"{label} {width}->{output_width}"
        # BF16 kernels may differ at rounding boundaries; FP8 is compared with its quantized operands
        check(run(operation(bf16_operation), x, weight, offsets, grad), expected_bf16, f"BF16 {suffix}", .006)
        check(run(operation(fp8_operation), x, weight, offsets, grad), expected_fp8, f"FP8 {suffix}", .001)
        if args.triton_wgrad:
            check(run(operation(fp8_triton_wgrad_operation), x, weight, offsets, grad), expected_fp8, f"FP8 Triton wgrad {suffix}", .001)

        if hasattr(grouped_fp8, "grouped_linear_fp8_relu2"):
            for wgrad_backend in wgrad_backends:
                def unfused(x, weight, offsets):
                    return grouped_fp8.grouped_linear_fp8(F.relu(x) ** 2, weight, offsets, wgrad_backend)

                def fused(x, weight, offsets):
                    return grouped_fp8.grouped_linear_fp8_relu2(x, weight, offsets, wgrad_backend)

                preactivation = x.bfloat16()
                expected_fused = run(operation(unfused), preactivation, weight, offsets, grad)
                actual_fused = run(operation(fused), preactivation, weight, offsets, grad)
                check(actual_fused, expected_fused, f"fused ReLU2 {wgrad_backend} {suffix}", .001)

        if grouped_mlp_fp8_relu2 is not None:
            down_weight = torch.randn(len(counts), width, output_width, device="cuda") * .03
            mlp_grad = torch.randn(sum(counts), width, device="cuda", dtype=torch.bfloat16) * .01
            for wgrad_backend in wgrad_backends:
                expected_mlp = run_mlp(operation(unfused_mlp), x, weight, down_weight, offsets, mlp_grad, wgrad_backend)
                actual_mlp = run_mlp(operation(fused_mlp), x, weight, down_weight, offsets, mlp_grad, wgrad_backend)
                check(actual_mlp, expected_mlp, f"producer fused MLP {wgrad_backend} {suffix}", .001)

print("PASS: grouped BF16/FP8 outputs and gradients, available fusion paths, empty/ragged experts, noncontiguous inputs, eager and fullgraph compilation")
