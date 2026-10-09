# 01 — native FP8 MLP / grouped MoE projections

An independent throughput and time-to-loss experiment against the current working tree. It changes projection arithmetic, so equal training loss per token is not assumed. The patch is unapplied. Applying it leaves the feature **off** until configured below.

```bash
git apply --check experiments/20261009-additional/01-fp8-projections.patch
git apply experiments/20261009-additional/01-fp8-projections.patch
```

Set these fields in `GPTConfig` before constructing the model:

```python
fp8_projections: str = "both"  # "up", "down", "both"; "off" restores original projections
fp8_min_features: int = 256
fp8_backward: bool = False
fp8_scaling: str = "dynamic"
fp8_margin: int = 0
```

`min_features` applies to both dimensions of each selected weight. An enabled configuration below that threshold still uses the original projection. The current default 2048↔4096 projections qualify. Both the normal MLP/expert loop and the **actual packed grouped MoE path** use the selected kernels. Router, biases, activations, attention projections, attention implementation, embeddings and LM head retain their existing arithmetic. Ragged dimensions, including fair SwiGLU widths, are masked to full tiles rather than requiring model-width changes.

## Backend and hardware

Use **PyTorch 2.10.0 with its CUDA 13.0 build and Triton 3.6.0**, on Linux with a CUDA-compatible driver. No package was installed. The code checks the exact PyTorch/Triton versions when enabled. PyTorch's [versioned dependency pin](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/.ci/docker/triton_version.txt) specifies Triton 3.6.0.

This patch intentionally accepts **SM89 (Ada)** and **SM120 (RTX 50 / RTX PRO Blackwell)** only. Other devices fail explicitly when enabled. The v3.6 [native-matmul selection](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp) identifies these architectures as native FP8 MMA-v2 targets. The selected operands are `tl.float8e4nv`, and [the corresponding lowering](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp) emits `mma.sync...f32.e4m3.e4m3.f32`. This is actual FP8 tensor-core multiplication, not BF16 multiplication of fake-quantized values. The optional checker below verifies the generated PTX; that check has **not** been run.

## Scaling and gradients

Dynamic E4M3 scaling uses `s = max(amax, 1e-12) * 2**fp8_margin / 448` and `q = E4M3(clamp(x/s, -448, 448))`. Products accumulate in FP32 and are multiplied by their two dequantization scales before casting to the original projection output dtype. There is one scale for the full packed activation tensor, one per expert weight matrix, and one for the full output-gradient tensor. Dense MLP weights are one expert. Activation and gradient scales therefore share outlier sensitivity across experts.

`fp8_scaling = "fixed"` instead uses `fp8_input_scale`, `fp8_weight_scale`, and `fp8_grad_scale` as positive dequantization scales. These are a shared input/weight/gradient triple for selected up/down projections, not per-layer calibrations. The fixed defaults `1.0` are uncalibrated; calibrate before comparing quality. The margin affects only dynamic scales. Fixed scaling omits the tensor-wide amax reductions and can clip if poorly calibrated.

With `fp8_backward = False` (the initial experiment), only selected forward GEMMs use FP8. Backward calls the original grouped BF16/FP32 implementation on saved original activation and weight tensors. This is an **approximate linear/surrogate gradient**, not the mathematical derivative of discrete FP8 rounding, and not an exact straight-through derivative using the quantized operands. With `True`, dgrad and wgrad also use native E4M3 FP8 GEMMs, independently quantizing the original saved tensors and the incoming gradient. Scaling/rounding has no gradient. E4M3 is used for gradients too; an E5M2 recipe is not implemented. Numerical stability and loss curves must determine whether that option is useful.

For dgrad, each expert's token interval uses its own weight matrix and weight scale. For wgrad, the reduction is restricted to that expert's token interval. Empty experts write zero weight gradients; their forward/dgrad intervals have no output rows. All-zero tensors have a positive scale and stay zero. Values below the amax floor can lose relative accuracy.

## State, compilation and memory

Original `nn.Linear` parameters, names, shapes and ordering remain intact, including separate 2D expert weights for Muon. Existing stacking, dtype conversion, optimizer updates and safetensors checkpoint flow remain in place. This adds no parameter, persistent quantized weight, scale buffer, calibration history or optimizer state. Weight state-dict shapes and names remain compatible with a newly constructed model, but the normal training resume path is stricter: `asdict(gpt_config)` automatically includes all new precision fields in `resume_config["model"]`. Pre-patch checkpoints therefore fail configuration equality even when the selector is `"off"`, and changing a precision setting also fails that check. Use a fresh run for each experiment; no checkpoint migration is included.

The custom ops use `torch.library.triton_op`, `wrap_triton` and registered autograd, as the baseline grouped kernel does. They contain no host reads of expert counts, graph-break decorators or mutable amax history. Existing `fullgraph=True`, FlexAttention, DDP/PP and non-reentrant whole-block checkpointing settings are preserved. Quantization is deterministic and recomputed on checkpoint replay. Full-model compilation, checkpoint replay, distributed execution and optimizer resume remain **unverified on GPU**.

Autograd saves the original `x`, stacked `weight` and `offsets`, not FP8 copies. Tile quantization is fused into the GEMM and repeated when tiles share operands. Packed FP8 weights/activations are not stored between calls. Consequently this experiment **does not reduce persistent model or saved-activation memory**. Stacked weights still occupy memory; dynamic `x.float().abs().amax(...)` can introduce large FP32 temporaries in eager execution, and compiler fusion/memory reuse must be measured. Conversions, reduction launches, and the deliberately fixed 128×64×64 tile can offset tensor-core gains or cause regressions. Native low precision is not a speedup claim.

## Verification and later experiment

Locally completed: independent apply check against the live tree, scratch apply, Python AST parsing and byte-exact scratch reversal; original production-file hashes unchanged. A scalar schedule audit covered 24 combinations of empty/ragged expert counts, worker counts and output-row/reduction grouping. There is no local PyTorch/CUDA installation, so kernel compilation, numerical GPU behavior, performance and training have not been tested.

After applying this patch, the optional checker can be run explicitly on the target GPU:

```bash
python experiments/20261009-additional/check_precision_projections.py --precision fp8
```

It checks eager, fullgraph compiled, and fullgraph compiled non-reentrant checkpoint forward/backward against a separate quantized PyTorch reference; tests empty/ragged experts, strided tensors, fixed/dynamic scaling, zero/tiny ranges, and both backward settings; and inspects PTX for native FP8 MMA. It is provided **unrun**, and does not run a training job. It does not establish full-model or convergence correctness.

For later authorized training runs, first compare `"off"` against forward-only `"up"`, `"down"` and `"both"`, using identical seeds/data/order/effective token batch and unchanged attention/checkpoint settings. Record warmed tokens/s, full-step latency, peak allocated/reserved memory, validation loss against tokens, and wall time to the same validation target. Treat native-backward and fixed-scale recipes as separate ablations. Include quantization and optimizer time in throughput. Reject configurations that improve GEMM timing while worsening time-to-loss or exceeding the current memory limit.

To remove the applied experiment, restore any edited precision config values to the patch defaults, then use `git apply -R experiments/20261009-additional/01-fp8-projections.patch`. The FP8 and NVFP4 files are independent alternatives against the same baseline; applying both together is not supported by these patches.
