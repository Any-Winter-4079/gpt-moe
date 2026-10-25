from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.linear.default.grouped import grouped_linear
from model.linear.fp8.custom.grouped import _quantize_expert_weights, grouped_linear_fp8_weights


def quantized_reference(x, block_rows):
    m, k = x.shape
    padded = F.pad(x.detach().float(), (0, (-k) % 128, 0, (-m) % block_rows))
    values = padded.reshape(padded.shape[0] // block_rows, block_rows, padded.shape[1] // 128, 128)
    amax = values.abs().amax((1, 3), keepdim=True)
    scale = torch.where(amax == 0, 1.0, torch.exp2(torch.ceil(torch.log2(amax / 448.0))).clamp_min(2.0**-126))
    return ((values / scale).to(torch.float8_e4m3fn).float() * scale).reshape(padded.shape)[:m, :k]


def fp8_reference(x, weight, grad, counts, relu2=False, order=None, top_k=1):
    source = x[order // top_k] if order is not None else x
    outputs, input_grads, weight_grads = [], [], []
    start = 0
    for expert, count in enumerate(counts):
        group_x = source[start:start + count]
        activation = F.relu(group_x) ** 2 if relu2 else group_x
        group_grad = grad[start:start + count]
        quantized_weight = quantized_reference(weight[expert], 128)
        outputs.append((quantized_reference(activation, 1) @ quantized_weight.t()).bfloat16())
        input_grad = quantized_reference(group_grad, 1) @ quantized_weight
        if relu2:
            # preserve the bf16 gradient and derivative boundaries of the unfused activation
            input_grad = input_grad.to(x.dtype) * (2 * group_x.clamp_min(0))
        input_grads.append(input_grad.to(x.dtype))
        weight_grads.append(quantized_reference(group_grad.t(), 1) @ quantized_reference(activation.t(), 1).t())
        start += count
    input_grad = torch.cat(input_grads)
    if order is not None:
        inverse_order = torch.argsort(order).view(x.shape[0], top_k)
        routed_grad = torch.zeros_like(x)
        for selected in range(top_k):
            routed_grad = (routed_grad.float() + input_grad[inverse_order[:, selected]].float()).to(x.dtype)
        input_grad = routed_grad
    return torch.cat(outputs), input_grad, torch.stack(weight_grads).to(weight.dtype)


def run(operation, x, weight, offsets, grad):
    x = x.detach().clone().requires_grad_()
    weights = [torch.nn.Parameter(value.detach().clone()) for value in weight.unbind(0)]
    output = operation(x, weights, offsets)
    output.backward(grad)
    assert output.dtype == grad.dtype and x.grad.dtype == x.dtype
    assert all(value.grad is not None and value.grad.dtype == value.dtype for value in weights)
    return output.detach(), x.grad.detach(), torch.stack([value.grad for value in weights])


def run_accumulated(operation, inputs, weights, offsets, grads, prepared_weight):
    for weight in weights:
        weight.grad = None
    outputs, input_grads = [], []
    for source, grad in zip(inputs, grads):
        x = source.detach().clone().requires_grad_()
        output = operation(x, weights, offsets, prepared_weight)
        (output / len(inputs)).backward(grad)
        assert output.dtype == torch.bfloat16 and x.grad.dtype == x.dtype
        outputs.append(output.detach())
        input_grads.append(x.grad.detach())
    assert all(weight.grad is not None and weight.grad.dtype == weight.dtype for weight in weights)
    return torch.cat(outputs), torch.cat(input_grads), torch.stack([weight.grad for weight in weights])


def check(actual, expected, label, tolerance):
    for expert, count in enumerate(counts):
        if count == 0:
            assert not actual[2][expert].count_nonzero(), f"{label}: empty expert {expert} received a weight gradient"
    for name, value, reference in zip(("output", "input gradient", "weight gradient"), actual, expected):
        assert value.shape == reference.shape and torch.isfinite(value).all(), (label, name)
        difference = (value.float() - reference.float()).norm()
        relative_error = difference / reference.float().norm().clamp_min(1e-12)
        print(f"{label} {name}: relative L2={relative_error.item():.8f}")
        assert relative_error < tolerance, (label, name, relative_error.item())


def default_operation(x, weights, offsets):
    return grouped_linear(x, torch.stack(weights), offsets)


def default_reference(x, weights, offsets):
    pieces, start = [], 0
    for weight, count in zip(weights, counts):
        pieces.append(F.linear(x[start:start + count], weight))
        start += count
    return torch.cat(pieces)


torch.manual_seed(2026)
torch.set_float32_matmul_precision("highest")
assert torch.cuda.is_available(), "this correctness check requires CUDA"
assert torch.cuda.get_device_capability() == (12, 0), "this correctness check requires SM120"
counts = [0, 1, 0, 128, 134, 0, 257, 0]
offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)

for width, output_width in ((128, 256), (260, 132)):
    x = torch.randn(width, sum(counts), device="cuda").t()
    x[:2] = 0
    weight = torch.randn(len(counts), output_width, width, device="cuda") * .03
    weight[3, :128, :128] = 0
    grad = torch.randn(sum(counts), output_width, device="cuda", dtype=torch.bfloat16) * .01
    assert not x.is_contiguous()

    for label, compile_operation in (("eager", lambda fn: fn), ("compiled", lambda fn: torch.compile(fn, fullgraph=True))):
        suffix = f"{label} {width}->{output_width}"
        for dtype, tolerance in ((torch.float32, .00001), (torch.float16, .002), (torch.bfloat16, .006)):
            expected = run(default_reference, x.to(dtype), weight.to(dtype), offsets, grad.to(dtype))
            actual = run(compile_operation(default_operation), x.to(dtype), weight.to(dtype), offsets, grad.to(dtype))
            check(actual, expected, f"default {dtype} {suffix}", tolerance)

        for relu2 in (False, True):
            def fp8_operation(x, weights, offsets):
                return grouped_linear_fp8_weights(x, weights, offsets, relu2=relu2)

            source = x.bfloat16() if relu2 else x
            expected = fp8_reference(source, weight, grad, counts, relu2)
            actual = run(compile_operation(fp8_operation), source, weight, offsets, grad)
            check(actual, expected, f"FP8 relu2={relu2} {suffix}", .001)

        for top_k in (1, 2):
            order = torch.randperm(sum(counts), device="cuda")
            inverse_order = torch.argsort(order)
            for dtype in (torch.float32, torch.bfloat16):
                if label == "compiled":
                    torch.compiler.reset()

                def routed(x, weights, offsets):
                    return grouped_linear_fp8_weights(
                        x, weights, offsets, order=order, top_k=top_k, inverse_order=inverse_order,
                    )

                source = torch.randn(width, sum(counts) // top_k, device="cuda", dtype=dtype).t()
                expected = fp8_reference(source, weight, grad, counts, order=order, top_k=top_k)
                actual = run(compile_operation(routed), source, weight, offsets, grad)
                check(actual, expected, f"FP8 routed top-k={top_k} {dtype} {suffix}", .001)

        for relu2, pack in ((False, False), (True, False), (False, True)):
            source = x.bfloat16() if relu2 else x
            if pack:
                source = torch.randn(width, sum(counts) // 2, device="cuda").t()
            order = torch.randperm(sum(counts), device="cuda") if pack else None

            def accumulated(x, weights, offsets, prepared_weight):
                return grouped_linear_fp8_weights(
                    x, weights, offsets, relu2=relu2, order=order, top_k=2 if pack else 1,
                    prepared_weight=prepared_weight,
                )

            if label == "compiled":
                torch.compiler.reset()
            accumulated = compile_operation(accumulated)
            inputs = [source + torch.randn_like(source) * .01 for _ in range(4)]
            grads = [grad + torch.randn_like(grad) * .001 for _ in range(4)]
            reference_weights = [torch.nn.Parameter(value.clone()) for value in weight]
            cached_weights = [torch.nn.Parameter(value.clone()) for value in weight]
            for update in range(2):
                with torch.no_grad():
                    prepared_weight = _quantize_expert_weights(cached_weights)
                expected = run_accumulated(accumulated, inputs, reference_weights, offsets, grads, None)
                actual = run_accumulated(accumulated, inputs, cached_weights, offsets, grads, prepared_weight)
                check(actual, expected, f"cached GA4 relu2={relu2} pack={pack} update={update} {suffix}", .001)
                with torch.no_grad():
                    for values in (reference_weights, cached_weights):
                        for value in values:
                            value.add_(.125)

print("PASS: grouped default/FP8 outputs and gradients, ReLU2 fusion, inverse routing, cached accumulation, empty/ragged experts, noncontiguous inputs, eager and fullgraph compilation")
