from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.linear.fp8.custom.dense import project, quantize_weights


def quantized_reference(x, block_rows):
    m, k = x.shape
    padded = F.pad(x.detach().float(), (0, (-k) % 128, 0, (-m) % block_rows))
    values = padded.reshape(-1, block_rows, padded.shape[1] // 128, 128)
    amax = values.abs().amax((1, 3), keepdim=True)
    scale = torch.where(amax == 0, 1.0, torch.exp2(torch.ceil(torch.log2(amax / 448.0))).clamp_min(2.0**-126))
    return ((values / scale).to(torch.float8_e4m3fn).float() * scale).reshape(padded.shape)[:m, :k]


def check(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all(), label
    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    print(f"{label}: relative L2={relative_error.item():.8f}")
    assert relative_error < .001, (label, relative_error.item())


torch.manual_seed(2026)
torch.set_float32_matmul_precision("highest")
assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0), "this correctness check requires SM120"

for width, output_widths in ((128, (128, 128, 128)), (256, (256, 64, 64)), (64, (256,)), (256, (128,))):
    for label, operation in (("eager", project), ("compiled", torch.compile(project, fullgraph=True))):
        torch.compiler.reset()
        for dtype in (torch.float32, torch.bfloat16):
            for cached in (False, True):
                torch.compiler.reset()
                weights = [torch.nn.Parameter(torch.randn(n, width, device="cuda") * .03) for n in output_widths]
                for update in range(2):
                    for weight in weights:
                        weight.grad = None
                    with torch.no_grad():
                        prepared = quantize_weights(weights) if cached else None
                    expected_weights = [torch.zeros_like(weight) for weight in weights]
                    for batch in (4, 8):
                        x = torch.randn(batch, width, 33, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
                        outputs = operation(x, weights, prepared)
                        assert all(value.stride(-1) == 1 for value in outputs)
                        grads = [torch.randn_like(value) * .01 for value in outputs]
                        torch.autograd.backward(outputs, grads)
                        source = x.detach().reshape(-1, width)
                        expected_x = torch.zeros_like(source, dtype=torch.float32)
                        for index, (weight, output, grad) in enumerate(zip(weights, outputs, grads)):
                            gradient = grad.reshape(-1, weight.shape[0])
                            quantized_weight = quantized_reference(weight, 128)
                            expected_out = (quantized_reference(source, 1) @ quantized_weight.t()).bfloat16()
                            expected_x += quantized_reference(gradient, 1) @ quantized_weight
                            expected_weights[index] += quantized_reference(gradient.t(), 1) @ quantized_reference(source.t(), 1).t()
                            check(output.reshape_as(expected_out), expected_out, f"{label} cached={cached} update={update} output {index}")
                            check(weight.grad, expected_weights[index], f"{label} cached={cached} update={update} weight {index}")
                        check(x.grad, expected_x.to(dtype).reshape_as(x), f"{label} cached={cached} update={update} input")
                    with torch.no_grad():
                        for weight in weights:
                            weight.add_(.01)
        if len(output_widths) == 3:
            weights = [torch.nn.Parameter(torch.randn(n, width, device="cuda") * .03) for n in output_widths]
            x = torch.randn(2, 17, width, device="cuda", requires_grad=True)
            operation(x, weights)[0].sum().backward()
            assert torch.isfinite(x.grad).all()
            assert not weights[1].grad.count_nonzero() and not weights[2].grad.count_nonzero()

print("PASS: dense/fused QKV FP8 outputs, input and accumulated weight gradients, cache refresh, padded GQA widths, changing batches, noncontiguous inputs and eager/fullgraph compilation")
