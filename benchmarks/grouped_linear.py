"""Run from the repository root: python -m benchmarks.grouped_linear --help."""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import time
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from config.model import GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare grouped BF16/FP8 expert projections and a PyTorch BF16 reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--projections", nargs="+", choices=["up", "down"], default=["up", "down"])
    parser.add_argument("--routing", choices=["random", "balanced"], default="random")
    parser.add_argument("--counts", nargs="+", type=int, help="actual tokens per expert, for one batch size")
    parser.add_argument("--iterations", type=int, default=20, help="calls per timed round")
    parser.add_argument("--repeats", type=int, default=5, help="rounds with rotated implementation order")
    parser.add_argument("--warmup", type=int, default=5, help="untimed calls, including compilation and autotuning")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eager", action="store_true", help="disable torch.compile for complete projection timings")
    args = parser.parse_args()
    if min(*args.batch_sizes, args.seq_len, args.iterations, args.repeats, args.warmup) < 1:
        parser.error("batch sizes, sequence length, iterations, repeats and warmup must be positive")
    if args.counts is not None and len(args.batch_sizes) != 1:
        parser.error("--counts requires exactly one batch size")
    return args


def expert_counts(args: argparse.Namespace, config: GPTConfig, batch_size: int) -> List[int]:
    tokens = batch_size * args.seq_len
    assignments = tokens * config.moe_top_k
    if args.counts is not None:
        counts = args.counts
        if (len(counts) != config.num_experts or sum(counts) != assignments
                or any(count < 0 or count > tokens for count in counts)):
            raise ValueError("counts must contain one value per expert, sum to tokens * top_k, and lie in [0, tokens]")
        return counts
    if args.routing == "balanced":
        return [assignments // config.num_experts + (i < assignments % config.num_experts)
                for i in range(config.num_experts)]
    # synthetic routing selects distinct experts per token and exercises unaligned boundaries
    generator = torch.Generator().manual_seed(args.seed)
    scores = torch.randn(tokens, config.num_experts, generator=generator)
    indices = scores.topk(config.moe_top_k, dim=-1).indices
    return torch.bincount(indices.flatten(), minlength=config.num_experts).tolist()


def projection_functions(offsets: Tensor, counts: List[int]) -> Dict[str, Callable]:
    from model.linear.grouped import grouped_linear
    from model.linear.grouped_fp8 import grouped_linear_fp8

    def bf16(x: Tensor, weight: Tensor) -> Tensor:
        return grouped_linear(x.to(torch.bfloat16), weight.to(torch.bfloat16), offsets)

    def fp8(x: Tensor, weight: Tensor) -> Tensor:
        return grouped_linear_fp8(x, weight, offsets)

    def reference(x: Tensor, weight: Tensor) -> Tensor:
        x = x.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        return torch.cat([F.linear(part, weight[expert])
                          for expert, part in enumerate(x.split(counts))], dim=0)

    return {"grouped_bf16": bf16, "grouped_fp8": fp8, "torch_bf16": reference}


def forward_backward(forward: Callable, x: Tensor, weight: Tensor, grad: Tensor) -> Callable:
    def run() -> Tuple[Tensor, Tensor, Tensor]:
        out = forward(x, weight)
        grad_x, grad_weight = torch.autograd.grad(out, (x, weight), grad)
        return out, grad_x, grad_weight

    return run


def measure(cases: Dict[str, Callable], args: argparse.Namespace) -> Dict[str, float]:
    for name, fn in cases.items():
        print(f"  warming {name}", flush=True)
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()

    samples = {name: [] for name in cases}
    wall_samples = {name: [] for name in cases}
    names = list(cases)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(len(names) * args.repeats)]
    # initialize event resources before measurement
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()

    for repeat in range(args.repeats):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for index, name in enumerate(order):
            start, end = events[repeat * len(names) + index]
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start.record()
            for _ in range(args.iterations):
                cases[name]()
            end.record()
            end.synchronize()
            wall_samples[name].append((time.perf_counter() - wall_start) * 1000 / args.iterations)
            samples[name].append(start.elapsed_time(end) / args.iterations)

    print("  median time per call; min/max are across round averages")
    print(f"  {'implementation / component':<30} {'CUDA ms':>10} {'min':>10} {'max':>10} {'wall ms':>10}")
    for name in names:
        values = samples[name]
        print(f"  {name:<30} {statistics.median(values):>10.4f} {min(values):>10.4f} "
              f"{max(values):>10.4f} {statistics.median(wall_samples[name]):>10.4f}")
    return {name: statistics.median(values) for name, values in samples.items()}


def compare_results(name: str, actual: Tuple[Tensor, ...], reference: Tuple[Tensor, ...]) -> None:
    print(f"  {name} error versus eager PyTorch BF16:")
    for label, value, expected in zip(("output", "input gradient", "weight gradient"), actual, reference):
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise AssertionError(f"{name} {label}: shape or dtype differs from the reference")
        value, expected = value.float(), expected.float()
        if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"{name} {label}: non-finite values")
        difference = value - expected
        relative_l2 = (torch.linalg.vector_norm(difference)
                       / torch.linalg.vector_norm(expected).clamp_min(1e-30)).item()
        max_abs = difference.abs().max().item()
        print(f"    {label:<16} relative L2={relative_l2:.6f}  max absolute={max_abs:.6g}")
        # allow reduction-order differences in bf16; report fp8 error without claiming convergence
        if name != "grouped_fp8" and relative_l2 > 0.02:
            raise AssertionError(f"{name} {label}: BF16 relative L2 error exceeds 2%")


def component_functions(x: Tensor, weight: Tensor, grad: Tensor, offsets: Tensor) -> Dict[str, Callable]:
    import triton
    from torch.library import wrap_triton

    from model.linear.grouped import _grouped_mm
    from model.linear.grouped_fp8 import _grouped_fp8_mm, _quantize_activations, _quantize_weights_kernel

    x, weight, grad = x.detach(), weight.detach(), grad.detach()
    experts, n, k = weight.shape

    def quantize_weight() -> Tuple[Tensor, Tensor]:
        # match the allocations and launch in grouped_fp8._forward
        quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
        scale = torch.empty((experts, triton.cdiv(n, 128), triton.cdiv(k, 128)),
                            device=weight.device, dtype=torch.float32)
        wrap_triton(_quantize_weights_kernel)[(experts, triton.cdiv(n, 128), triton.cdiv(k, 128))](
            weight, quantized, scale, n, k, *weight.stride(), num_warps=8,
        )
        return quantized, scale

    x_bf16, weight_bf16 = x.to(torch.bfloat16), weight.to(torch.bfloat16)
    x_row, x_scale, x_column, x_column_scale = _quantize_activations(x, offsets, True)
    grad_row, grad_scale, grad_column, grad_column_scale = _quantize_activations(grad, offsets, True)
    weight_fp8, weight_scale = quantize_weight()

    return {
        "bf16_cast_input_weight": lambda: (x.to(torch.bfloat16), weight.to(torch.bfloat16)),
        "bf16_forward_gemm": lambda: _grouped_mm(x_bf16, weight_bf16.transpose(-2, -1), offsets),
        "bf16_input_grad_gemm": lambda: _grouped_mm(grad, weight_bf16, offsets),
        "bf16_weight_grad_gemm": lambda: _grouped_mm(grad.t(), x_bf16, offsets),
        "fp8_quant_input_row_column": lambda: _quantize_activations(x, offsets, True),
        "fp8_quant_weight": quantize_weight,
        "fp8_quant_grad_row_column": lambda: _quantize_activations(grad, offsets, True),
        "fp8_forward_gemm": lambda: _grouped_fp8_mm(
            x_row, weight_fp8.transpose(-2, -1), x_scale, weight_scale.transpose(-2, -1), offsets, torch.bfloat16,
        ),
        "fp8_input_grad_gemm": lambda: _grouped_fp8_mm(
            grad_row, weight_fp8, grad_scale, weight_scale, offsets, x.dtype,
        ),
        "fp8_weight_grad_gemm": lambda: _grouped_fp8_mm(
            grad_column, x_column.t(), grad_column_scale.t(), x_column_scale, offsets, weight.dtype,
        ),
    }


def benchmark_projection(
    args: argparse.Namespace, config: GPTConfig, batch_size: int, projection: str, counts: List[int],
) -> None:
    hidden_dim = config.mlp_hidden_dim
    if config.activation == "swiglu" and config.use_fair_swiglu:
        hidden_dim = round((2.0 / 3.0) * hidden_dim)
    k, n = ((config.d_model, hidden_dim * (2 if config.activation == "swiglu" else 1))
            if projection == "up" else (hidden_dim, config.d_model))
    m = sum(counts)
    print(f"\nbatch={batch_size}, projection={projection}, packed tokens={m}, "
          f"input features={k}, output features={n}, experts={config.num_experts}", flush=True)
    print(f"  tokens per expert: {counts}")
    torch.manual_seed(args.seed)
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = (torch.randn(config.num_experts, n, k, device="cuda") / math.sqrt(k)).requires_grad_()
    grad = (torch.randn(m, n, device="cuda") / math.sqrt(n)).to(torch.bfloat16)
    functions = projection_functions(offsets, counts)

    print("  checking output and both gradients", flush=True)
    reference = forward_backward(functions["torch_bf16"], x, weight, grad)()
    cases = {}
    for name, forward in functions.items():
        if not args.eager:
            forward = torch.compile(forward, mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
        run = forward_backward(forward, x, weight, grad)
        print(f"  preparing {name} (first call may compile/autotune)", flush=True)
        actual = run()
        compare_results(name, actual, reference)
        del actual
        cases[name] = run
    del reference

    print("  complete projection: forward + backward, including casts/quantization", flush=True)
    timings = measure(cases, args)
    print(f"  grouped FP8 / grouped BF16 time: {timings['grouped_fp8'] / timings['grouped_bf16']:.3f}x "
          "(below 1 is faster)")
    del cases

    print("  isolated components: eager launches with prepared operands; GEMMs include block rescaling", flush=True)
    print("  component times are diagnostic and should not be added to predict compiled totals")
    components = component_functions(x, weight, grad, offsets)
    measure(components, args)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires an NVIDIA CUDA GPU and the training image dependencies")
    import triton

    config = GPTConfig()
    torch.set_float32_matmul_precision("high")
    device = torch.cuda.get_device_properties(0)
    print(f"GPU: {device.name}, compute capability {device.major}.{device.minor}, "
          f"{device.total_memory / 2**30:.1f} GiB", flush=True)
    print(f"Python: {platform.python_version()}, PyTorch: {torch.__version__}, "
          f"CUDA: {torch.version.cuda}, Triton: {triton.__version__}")
    print(f"Mode: {'eager' if args.eager else 'torch.compile max-autotune-no-cudagraphs, fullgraph=True'}")
    print(f"Timing: {args.repeats} rounds x {args.iterations} calls; {args.warmup} warmup calls per case")
    print("Shapes: GPTConfig expert widths/count/top-k; batch sizes and sequence length from command-line options")
    print("Inputs: fixed synthetic Gaussian FP32 activations/weights and BF16 output gradients, reused across implementations")
    print("Routing: supplied counts" if args.counts is not None else f"Routing: synthetic {args.routing} counts")
    print("Scope: one packed projection and its gradients; includes fresh weight quantization on every call")
    print("Excludes routing/packing, weight stacking, activation functions, optimizer updates and gradient accumulation")
    print("CUDA event intervals can include launch gaps; wall times also include host overhead and synchronization")
    print("FP8 numerical errors are reported, not treated as evidence of unchanged training convergence")

    for batch_size in args.batch_sizes:
        counts = expert_counts(args, config, batch_size)
        for projection in args.projections:
            benchmark_projection(args, config, batch_size, projection, counts)
            torch.cuda.synchronize()
            torch._dynamo.reset()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
