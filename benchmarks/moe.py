"""Run from the repository root: python -m benchmarks.moe --help."""

from __future__ import annotations

import argparse
import copy
import math
import platform
import statistics
import time
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor, nn

from config.model import GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare complete MoE forward/backward with fresh Gaussian inputs and live top-k routing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=20, help="different inputs per round")
    parser.add_argument("--repeats", type=int, default=5, help="rounds; inputs are not repeated between rounds")
    parser.add_argument("--warmup", type=int, default=5, help="different untimed inputs per implementation")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eager", action="store_true", help="disable torch.compile for all implementations")
    args = parser.parse_args()
    if min(*args.batch_sizes, args.seq_len, args.iterations, args.repeats, args.warmup) < 1:
        parser.error("batch sizes, sequence length, iterations, repeats and warmup must be positive")
    return args


@torch.compiler.disable
def split_expert_inputs(packed: Tensor, counts: Tensor) -> Tuple[Tensor, ...]:
    # keep the GPU-to-CPU synchronization inside the measured forward pass
    return packed.split(counts.cpu().tolist())


def torch_forward_grouped(
    self: nn.Module, hidden_states: Tensor, top_k_indices: Tensor, top_k_weights: Tensor,
) -> Tensor:
    expert_indices, order = top_k_indices.reshape(-1).sort()
    counts = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.int32).scatter_add_(
        0, expert_indices, torch.ones_like(expert_indices, dtype=torch.int32),
    )
    packed = hidden_states[order // self.top_k]
    parts = split_expert_inputs(packed, counts)
    packed = torch.cat([expert(part) for expert, part in zip(self.experts, parts)], dim=0)
    inverse_order = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=order.device))
    expert_output = packed[inverse_order].view(hidden_states.shape[0], self.top_k, -1).to(hidden_states.dtype)
    return (expert_output * top_k_weights[:, :, None].to(hidden_states.dtype)).sum(dim=1)


def build_models(config: GPTConfig, seed: int) -> Dict[str, nn.Module]:
    from config.training import TrainingConfig
    from model.feedforward.moe import MoE

    class TorchMoE(MoE):
        forward_grouped = torch_forward_grouped

    training_config = TrainingConfig()
    training_config.use_bf16_autocast = True
    config = copy.copy(config)
    config.use_grouped_moe = True
    torch.manual_seed(seed)
    bf16 = MoE(config, training_config, config.num_experts, config.moe_top_k).cuda().train()
    fp8 = MoE(config, training_config, config.num_experts, config.moe_top_k).cuda().train()
    fp8.load_state_dict(bf16.state_dict())
    for expert in fp8.experts:
        expert.up_proj.precision = "fp8"
        expert.down_proj.precision = "fp8"
    reference = TorchMoE(config, training_config, config.num_experts, config.moe_top_k).cuda().train()
    reference.load_state_dict(bf16.state_dict())
    return {"grouped_bf16": bf16, "grouped_fp8": fp8, "torch_packed_bf16": reference}


def make_step(model: nn.Module, forward: Callable, device_type: str = "cuda") -> Callable:
    parameters = tuple(model.parameters())
    balance_grad = parameters[0].new_tensor(model.load_balance_weight)

    def run(x: Tensor, grad: Tensor) -> Tuple[Tensor, ...]:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out, balance_loss = forward(x)
        if balance_loss is None:
            grads = torch.autograd.grad(out, (x, *parameters), grad)
        else:
            grads = torch.autograd.grad((out, balance_loss), (x, *parameters), (grad, balance_grad))
        return (out.detach(), *grads)

    return run


def random_inputs(shape: Tuple[int, ...], generator: torch.Generator, device: str) -> Tuple[Tensor, Tensor]:
    x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator).requires_grad_()
    grad = torch.randn(shape, device=device, dtype=torch.float32, generator=generator) / math.sqrt(shape[-1])
    return x, grad


@torch.no_grad()
def reference_counts(model: nn.Module, x: Tensor) -> List[int]:
    # diagnostic routing runs outside timing; measured forwards run their own router
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
        logits = model.router(x.reshape(-1, x.shape[-1])).float()
        indices = logits.topk(model.top_k, dim=-1).indices
    return torch.bincount(indices.flatten(), minlength=model.num_experts).cpu().tolist()


def metric_groups(model: nn.Module) -> Dict[str, List[int]]:
    groups = {"output": [0], "input gradient": [1], "router gradients": [], "expert gradients": []}
    for i, (name, _) in enumerate(model.named_parameters()):
        groups["router gradients" if name.startswith("router.") else "expert gradients"].append(i + 2)
    return groups


@torch.no_grad()
def error_metrics(
    actual: Tuple[Tensor, ...], reference: Tuple[Tensor, ...], groups: Dict[str, List[int]],
) -> Tensor:
    if len(actual) != len(reference):
        raise AssertionError("output/gradient count differs from the reference")
    stats = []
    for indices in groups.values():
        difference_squared = actual[0].new_zeros((), dtype=torch.float32)
        reference_squared = difference_squared.clone()
        maximum = difference_squared.clone()
        for i in indices:
            value, expected = actual[i], reference[i]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise AssertionError("output/gradient shape or dtype differs from the reference")
            value, expected = value.float(), expected.float()
            difference = value - expected
            difference_squared += difference.square().sum()
            reference_squared += expected.square().sum()
            maximum = torch.maximum(maximum, difference.abs().max())
        stats.append(torch.stack((difference_squared, reference_squared, maximum)))
    stats = torch.stack(stats).cpu().double()
    if not torch.isfinite(stats).all():
        raise AssertionError("non-finite output, gradient or error metric")
    return stats


def benchmark_batch(args: argparse.Namespace, config: GPTConfig, batch_size: int) -> None:
    from torch._dynamo.utils import counters

    print(f"\nbatch={batch_size}, sequence={args.seq_len}, width={config.d_model}, "
          f"experts={config.num_experts}, top_k={config.moe_top_k}", flush=True)
    models = build_models(config, args.seed)
    print(f"activation={config.activation}, balance loss weight={models['grouped_bf16'].load_balance_weight}")
    steps = {}
    for name, model in models.items():
        forward = model
        if not args.eager:
            # the PyTorch path splits by live CPU counts, then compiles variable-sized expert inputs
            is_reference = name == "torch_packed_bf16"
            forward = torch.compile(model, mode="max-autotune-no-cudagraphs",
                                    fullgraph=not is_reference, dynamic=is_reference)
        steps[name] = make_step(model, forward)
    eager_reference = make_step(models["torch_packed_bf16"], models["torch_packed_bf16"])
    groups = metric_groups(models["torch_packed_bf16"])
    shape = (batch_size, args.seq_len, config.d_model)
    generator = torch.Generator(device="cuda").manual_seed(args.seed + batch_size)

    print("warming all implementations on identical, changing inputs; compilation is untimed", flush=True)
    for warmup in range(args.warmup):
        x, grad = random_inputs(shape, generator, "cuda")
        expected = eager_reference(x, grad)
        for name, step in steps.items():
            print(f"  warmup {warmup + 1}/{args.warmup}: {name}", flush=True)
            actual = step(x, grad)
            stats = error_metrics(actual, expected, groups)
            if warmup == 0:
                relative = (stats[:, 0] / stats[:, 1].clamp_min(1e-30)).sqrt().tolist()
                print(f"    relative L2 versus eager reference: {dict(zip(groups, relative))}")
            del actual
        del expected, x, grad
    torch.cuda.synchronize()

    names = list(steps)
    cuda_times = {name: [] for name in names}
    wall_times = {name: [] for name in names}
    new_graphs = {name: 0 for name in names}
    errors = {name: torch.zeros(len(groups), 3, dtype=torch.float64) for name in names}
    counts_seen = []
    events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for name in names}
    for start, end in events.values():
        start.record()
        end.record()
    torch.cuda.synchronize()

    for repeat in range(args.repeats):
        for iteration in range(args.iterations):
            # random generation and diagnostics finish before any timing event starts
            x, grad = random_inputs(shape, generator, "cuda")
            counts_seen.append(reference_counts(models["torch_packed_bf16"], x))
            torch.cuda.synchronize()
            shift = (repeat + iteration) % len(names)
            order = names[shift:] + names[:shift]
            results = {}
            for name in order:
                start, end = events[name]
                graphs_before = counters["stats"]["unique_graphs"]
                wall_start = time.perf_counter()
                start.record()
                results[name] = steps[name](x, grad)
                end.record()
                end.synchronize()
                wall_times[name].append((time.perf_counter() - wall_start) * 1000)
                cuda_times[name].append(start.elapsed_time(end))
                new_graphs[name] += counters["stats"]["unique_graphs"] - graphs_before

            # check every measured input and all parameter gradients, outside the timed regions
            expected = eager_reference(x, grad)
            for name in names:
                stats = error_metrics(results[name], expected, groups)
                errors[name][:, :2] += stats[:, :2]
                errors[name][:, 2] = torch.maximum(errors[name][:, 2], stats[:, 2])
            del results, expected, x, grad
        print(f"  completed round {repeat + 1}/{args.repeats}", flush=True)

    print("timings per complete MoE forward/backward; new compilation during measurement is included")
    print(f"{'implementation':<22} {'CUDA median':>12} {'CUDA p95':>12} "
          f"{'wall mean':>12} {'wall max':>12} {'new graphs':>11}")
    for name in names:
        values = sorted(cuda_times[name])
        p95 = values[math.ceil(0.95 * len(values)) - 1]
        print(f"{name:<22} {statistics.median(values):>12.4f} {p95:>12.4f} "
              f"{statistics.mean(wall_times[name]):>12.4f} {max(wall_times[name]):>12.4f} {new_graphs[name]:>11}")
    print("times are milliseconds; new graphs counts Dynamo graph captures, including recompilations")

    print("reference-router token counts across measured inputs:")
    for expert, values in enumerate(zip(*counts_seen)):
        print(f"  expert {expert}: min={min(values)}, mean={statistics.mean(values):.1f}, max={max(values)}")
    print(f"  distinct count vectors: {len(set(tuple(counts) for counts in counts_seen))}/{len(counts_seen)}")
    print("numerical errors across all measured inputs versus eager PyTorch BF16:")
    for name in names:
        print(f"  {name}")
        for label, (difference_squared, reference_squared, maximum) in zip(groups, errors[name].tolist()):
            relative = math.sqrt(difference_squared / max(reference_squared, 1e-30))
            print(f"    {label:<18} relative L2={relative:.6f}  max absolute={maximum:.6g}")


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
    print(f"Timing: {args.repeats} rounds x {args.iterations} fresh inputs; {args.warmup} untimed warmup inputs")
    print("Fresh Gaussian FP32 inputs/upstream gradients on every call; identical input sequence and fixed weights for all candidates")
    print("Actual MoE router, top-k selection, packing, up/activation/down, weighted combination and backward")
    print("BF16 autocast; FP32 parameters; FP8 applies only to the grouped expert projections")
    print("PyTorch uses live counts copied to CPU inside forward, with per-expert MLP calls and packed output combination")
    print("Mode: eager" if args.eager else
          "Mode: max-autotune-no-cudagraphs; grouped fullgraph/static shapes, PyTorch partial graph/dynamic shapes")
    print("Random generation and numerical checks are untimed; routing/synchronization and any later compilation are timed")
    print("No optimizer updates or cross-microbatch gradient accumulation; synthetic activations do not establish training convergence")
    for batch_size in args.batch_sizes:
        benchmark_batch(args, config, batch_size)
        torch.cuda.synchronize()
        torch._dynamo.reset()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
