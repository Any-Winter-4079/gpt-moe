# 02 — native NVFP4 MLP / grouped MoE projections

An independent throughput and time-to-loss experiment against the current working tree. This is a simple deterministic NVFP4 arithmetic experiment, not NVIDIA's complete NVFP4 pretraining recipe. The patch is unapplied. Applying it leaves the feature **off** until configured below.

```bash
git apply --check experiments/20261009-additional/02-nvfp4-projections.patch
git apply experiments/20261009-additional/02-nvfp4-projections.patch
```

Set these fields in `GPTConfig` before constructing the model:

```python
nvfp4_projections: str = "both"  # "up", "down", "both"; "off" restores original projections
nvfp4_min_features: int = 256
nvfp4_backward: bool = False
nvfp4_scaling: str = "dynamic"
nvfp4_margin: int = 0
```

The threshold applies to both dimensions of each selected projection. Current 2048↔4096 default projections qualify. Selected operations in both dense MLPs/expert loops and the **actual packed grouped MoE path** use the new backend. Router, biases, activations, all attention behavior, embeddings and LM head keep their existing arithmetic. Logical dimensions need not be multiples of 16: masked zeros complete the final reduction block without changing model widths.

## Concrete native backend

Use **PyTorch 2.10.0 with its CUDA 13.0 build and Triton 3.6.0**, on Linux with a CUDA-compatible driver. No dependency was installed. PyTorch's [versioned Triton pin](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/.ci/docker/triton_version.txt) is 3.6.0. The patch checks those exact PyTorch/Triton versions when enabled.

The supported target for this patch is **SM120**, including RTX 50-series and RTX PRO Blackwell. It explicitly rejects other devices; the SM100/SM103 data-center path is not claimed. This is useful here because [Transformer Engine's published NVFP4 training support](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) is narrower than all Blackwell GPU variants.

Triton v3.6.0's [SM120 scaled-matmul rewrite](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp) accepts packed E2M1 operands and scales. Its [MMA lowering](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp) selects `mma.sync...kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` when the scale tensors have E4M3 type. The implementation passes that type and an explicit scale for both operands. It does not ask an older Triton release to emulate unsupported block-scaled GEMMs in BF16. The optional checker inspects the generated PTX for this instruction family; compilation and that check are **unrun**.

## Format, scale configuration and rounding

Each operand is represented as E2M1 FP4 values, **one positive FP8 E4M3 scale per 16 reduction elements**, and an FP32 tensor dequantization scale. This is NVFP4, not 32-element/E8M0 MXFP4.

Dynamic tensor scales are `s = max(amax, 1e-12) * 2**nvfp4_margin / (6 * 448)`. There is one `s` for packed activations, one per expert weight matrix, and one for the complete output-gradient tensor. Within each row's 16-element reduction block, `b = E4M3(clamp(block_amax / (6*s), 2**-9, 448))`; FP4 values represent `x / (s*b)`. Using the rounded block scale in that division matches the scale supplied to the tensor cores. The final FP32 accumulator is multiplied by both tensor scales before returning the original output dtype.

`nvfp4_scaling = "fixed"` uses `nvfp4_input_scale`, `nvfp4_weight_scale` and `nvfp4_grad_scale` instead of the tensor amax reductions. These fixed FP32 tensor scales are shared across selected layers/projections and experts; the per-16-element E4M3 block scales are still computed dynamically. Fixed defaults `1.0` are uncalibrated, not recommended calibration values. The margin affects dynamic tensor scales only. Very large fixed scales can underflow local scales; very small ones saturate the representable range. The positive scale floors avoid division by zero, but intentionally sacrifice relative accuracy for sufficiently tiny values.

Packing uses native `cvt.rn.satfinite.e2m1x2.f32`, with the first tensor element in the low nibble. [NVIDIA's PTX packing definition](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt) places the second source argument in the low nibble, so the inline PTX passes `(hi, lo)`, matching [Triton v3.6.0’s packing implementation](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py). LHS packed shape is `(M, K/2)` and RHS packed shape `(K/2, N)`. Scale shapes are `(M, K/16)` and `(N, K/16)`; the RHS scale is deliberately **not transposed**, matching the [v3.6.0 block-scaled example](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/10-block-scaled-matmul.py).

This patch uses round-to-nearest-even and saturation, without random Hadamard transforms, stochastic rounding, or 2D weight scaling. Therefore it must not be described as reproducing the quality claims of [NVIDIA's NVFP4 pretraining work](https://arxiv.org/abs/2509.25149).

## Backward, empty experts and state

`nvfp4_backward = False` initially changes selected forward GEMMs only. The original grouped BF16/FP32 backward receives original saved `x` and `weight`. This is an **approximate linear/surrogate gradient**, not the derivative of the discrete quantizer, and not an exact straight-through derivative using quantized forward operands. With `True`, native NVFP4 computes dgrad and wgrad too, freshly quantizing along each GEMM's reduction dimension. Block scaling must be recomputed after changing orientation; this patch does so inside each GEMM instead of transposing packed bytes. Quantization and scale calculations have no gradients.

Expert offsets stay on GPU. Forward/dgrad partition output rows; wgrad partitions its token reduction. Each expert's wgrad reduction starts a fresh group of 16, and ragged tail lanes are zero-masked, so the quantizer does not mix values from adjacent experts. Empty experts have no forward/dgrad rows and produce zero wgrad. Strides are passed explicitly for activation/weight transposes. All-zero tensors use positive scales and remain zero.

The existing separate 2D trainable expert parameters, state-dict keys, shapes, ordering, initialization, Muon/AdamW state and safetensors checkpoint flow remain intact. No persistent quantized weights, scale state or new trainable parameters are added. Weight state-dict shapes and names remain compatible with a newly constructed model, but `asdict(gpt_config)` automatically includes the new precision fields in the strict `resume_config["model"]` comparison. Pre-patch checkpoints fail that check even with the selector `"off"`; changing precision settings also prevents normal resume. Use fresh runs for these comparisons. No checkpoint migration is included.

## Compilation, checkpointing and memory limits

The implementation uses the same registered `triton_op`/`wrap_triton`/autograd approach as the current grouped backend. It does not change `fullgraph=True`, disable compilation, synchronize expert counts to the CPU, or alter attention and whole-block non-reentrant checkpointing. Quantization is deterministic and has no history/RNG state to reconcile on replay. The complete compiled model, checkpoint recomputation, DDP/PP and resume behavior are still **unverified on GPU**.

Autograd retains original `x`, stacked `weight` and `offsets`, not FP4 tensors. FP4 packing and local-scale reduction occur inside each GEMM tile; no packed global tensor cache is introduced. Original BF16/FP32 source buffers and expert stacks remain. Dynamic tensor amax operations can create large FP32 intermediates before compilation/fusion, and the fused tile quantizer adds registers, local reductions and possible spills. Operands shared by multiple output tiles are quantized repeatedly. The fixed 128×64×64 launch is a starting point, not a tuned SM120 result. Do not expect lower peak model/activation memory merely because the GEMMs use FP4; this can be slower or use more memory than the baseline.

## Checks and later measurements

Completed locally: independent live-tree apply check, scratch apply/AST parsing/exact reversal, and unchanged production-file hashes. Pure scalar checks exercised 24 persistent scheduling combinations with empty/ragged experts, and 35 NVFP4 format/padding cases covering zeros, tiny ranges and nearest-even ties. These checks do not compile or execute Triton. PyTorch and CUDA are unavailable locally; runtime correctness, throughput and loss quality remain unmeasured.

After applying this patch, the optional GPU checker can be explicitly run:

```bash
python experiments/20261009-additional/check_precision_projections.py --precision nvfp4
```

It is provided **unrun**. It checks eager, fullgraph compiled and fullgraph compiled non-reentrant checkpoint forward/backward against an independent PyTorch quantized reference, both backward settings, strided inputs/weights, ragged/empty experts, fixed/dynamic scales, zero/tiny values, and native NVFP4 PTX. A failure must be resolved before training; passing it does not establish full-model correctness or convergence.

For later authorized experiments, compare `"off"` with forward-only `"up"`, `"down"`, then `"both"`, using identical data/seed/effective token batch and attention/checkpoint settings. Compare validation loss versus both tokens and elapsed time, warmed end-to-end tokens/s, step latency and peak allocated/reserved memory. Then test native backward and fixed scales as separate ablations if the forward-only run is useful. Include scale computation, quantization and optimizer time. A tensor-core instruction or lower GEMM latency alone does not establish faster time-to-loss.

To reverse, restore any modified precision configuration fields to the patch defaults and run `git apply -R experiments/20261009-additional/02-nvfp4-projections.patch`. FP8 and NVFP4 are independent alternatives against the same baseline and are not intended to stack directly.
