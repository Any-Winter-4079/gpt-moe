# Experimental patches

These patches target production code at commit `40af18c6ded4916a59b7e145ada749e459370218`. The unpatched baseline trains expert up/down projections and the LM head in FP8, with Triton forward/input-gradient expert kernels and DeepGEMM weight gradients. Validation, evaluation and sampling inherit the ordinary precision context.

Each patch applies independently to that baseline, except where a dependency is stated. Reverse the previous experiment before applying another; these are alternatives, not a patch series to apply together. The image and documentation additions do not change the production baseline.

Start with 012 for a small FP8 change that retains the quantization recipe, followed by 014 and 017 to test compact scales and a larger worker pool. Then compare 000/001 for the BF16 library path, 000/018 for BF16 scheduling, and 002 for the FP8 library path. The larger fusion experiments follow as 005, then 009 and 013. The remaining head, optimizer and RoPE experiments can be tested separately. The numbers identify patches rather than requiring every experiment to be run.

## Image

Build the separate [CUDA 13.4 DeepGEMM image](../README.md#build-and-push-the-experimental-cuda-134-deepgemm-image) before starting a fresh instance. It contains the pinned library, compiler, headers, and both SM120 API patches. The FP32-output API patch exposes support already present in the SM120 kernel; it does not change the matrix arithmetic of the baseline weight-gradient call.

Use the same GPU power limit for every comparison. The October 19 reference runs used 450 W; their times are not directly comparable with the earlier 600 W runs.

## Apply, run and reverse

Run from the repository root. Start with no edits to the files touched by the chosen patch:

```bash
experiment_patch=experiments/012-native-scaled-fp8.patch
git apply --check "$experiment_patch"
git apply "$experiment_patch"
git diff --stat
```

For grouped projection changes, run the optional CUDA correctness check before training:

```bash
python experiments/check_grouped.py
```

Then use the regular training entry point:

```bash
set -o pipefail
torchrun --standalone --nproc_per_node=1 train.py 2>&1 | tee training-012.log
```

After saving the results, reverse that same patch:

```bash
git apply --reverse --check "$experiment_patch"
git apply --reverse "$experiment_patch"
```

Reversal requires that the patched files have not been edited further. Check `git diff` before switching experiments. The code embedded in each training log records the applied implementation even when the working tree is uncommitted.

## Experiments

| Patch | Change | Compare with |
|---|---|---|
| `000-bf16-expert-control.patch` | Ordinary grouped BF16 expert projections; FP8 head | Control for 001 |
| `001-deepgemm-bf16-grouped.patch` | DeepGEMM BF16 forward, input-gradient and weight-gradient expert GEMMs | 000 |
| `002-deepgemm-fp8-projections.patch` | DeepGEMM FP8 expert forward and input-gradient GEMMs | Unpatched baseline |
| `003-deepgemm-quantization.patch` | DeepGEMM's helper for expert weight quantization | Unpatched baseline |
| `004-deepgemm-lm-head.patch` | DeepGEMM forward/backward for the FP8 LM head | Unpatched baseline |
| `005-sm120-fused-activation-quantization.patch` | Fuse ReLU² into down-projection FP8 preparation and its derivative into input-gradient stores | Unpatched baseline |
| `006-muon-foreach.patch` | Batch Muon momentum preparation and weight updates | Unpatched baseline |
| `007-muon-row-energy.patch` | Experiment with row activity weighting after Muon's polar projection | Unpatched baseline |
| `008-attention-strided-rope.patch` | One stride-aware Triton launch for Q/K RoPE and its backward | Unpatched baseline |
| `009-sm120-producer-epilogue-fusion.patch` | Produce the down projection's quantized inputs in the up-projection epilogue | 005; requires 005 |
| `010-head-fp32-logits.patch` | Produce FP32 logits directly from the DeepGEMM head | 004; requires 004 |
| `011-polar-fused-update.patch` | Fuse the matrix product and scaled residual in Polar Express | Unpatched baseline |
| `012-native-scaled-fp8.patch` | Use SM120's native block-scaled FP8 instructions with the existing quantization | Unpatched baseline |
| `013-native-scaled-producer-fusion.patch` | Use native scaled FP8 in the producer-fused MLP | 005 + 009; requires both |
| `014-native-scaled-uint8-scales.patch` | Store native scale exponents as bytes | 012; requires 012 |
| `015-native-producer-autotune.patch` | Autotune two producer launch configurations | 005 + 009 + 013; requires all three |
| `016-compact-scales-producer.patch` | Use compact scale bytes throughout the fused producer branch | 005 + 009 + 013 + 015; requires all four |
| `017-double-native-workers.patch` | Double the persistent native FP8 worker pool | 012 + 014; requires both |
| `018-double-bf16-workers.patch` | Double the ordinary BF16 grouped worker pool | 000; requires 000 for the BF16 control |
| `019-fused-pack-quantization.patch` | Gather routed tokens while preparing the up projection's FP8 operands | 005; requires 005 |
| `020-direct-expert-weight-quantization.patch` | Quantize expert parameters without stacking their FP32 weights | 005; requires 005 |
| `021-reuse-quantized-expert-weights.patch` | Reuse detached FP8 expert weights across accumulation microsteps | 005; requires 005 |
| `022-fused-pack-direct-weight-quantization.patch` | Combine routed input quantization with direct expert-weight quantization | 005 + 019; requires both and incorporates 020 |

### 000 and 001: BF16 expert projections

000 changes only the training precision of expert up/down projections to `default`, which uses BF16 under the current autocast configuration. The head remains FP8. Run this control and reverse it before applying 001.

001 makes the same precision change and replaces all three grouped BF16 GEMMs with DeepGEMM. FP32 master parameters, the existing BF16 cast boundaries, and routing remain intact. It prepares 128-row expert padding and GPU offsets without copying counts to the CPU. DeepGEMM's weight-gradient wrapper internally transposes both padded operands; that cost is part of the experiment.

### 002: FP8 expert forward and input gradients

This completes the DeepGEMM GEMM integration for the grouped FP8 expert projections. It retains E4M3, 1x128 activation/gradient blocks, and 128x128 weight blocks. Quantization directly writes the padded native operands, packed scales and transposed weights. A GPU unpack kernel follows each projection, and the transposed FP8 weights add temporary storage.

The patch adds `grouped_fp8_projection_backend = "deepgemm"`. Setting it to `"triton"` restores the original forward/input-gradient kernels within the experiment. The existing weight-gradient backend remains independent. The image's additional SM120 FP32-output API patch is required to retain FP32 input gradients.

### 003: library weight quantization

This replaces only expert weight preparation with `deep_gemm.utils.per_block_cast_to_fp8`, applied separately to each expert. Activation/gradient quantization and all GEMM calls stay as in the baseline. The helper clamps block maxima to at least `1e-4`; tiny blocks can therefore use different scales. Its bitwise power-of-two rounding can also differ at thresholds. These are part of the actual library behavior being tested.

The helper uses ordinary PyTorch operations that the existing compiled forward can trace. DeepGEMM becomes an import-time dependency when this patch is applied. The hypothesis is about the complete compiled preparation, not an assumption that the helper is faster.

### 004: DeepGEMM LM head

Only the FP8 `lm_head` path changes; other FP8 linears retain Transformer Engine. It uses the existing blockwise preparation, dense DeepGEMM forward/input-gradient GEMMs, and the existing weight-gradient integration with one group. The tied embedding parameter remains the same object. Under BF16 autocast, output and backward GEMM results retain Transformer Engine's BF16 rounding boundaries before gradients are returned to their original parameter/input dtype.

The patch also adds a CUDA correctness check for eager/full-graph execution, dequantized numerical references, and tied embedding/head gradients:

```bash
python experiments/check_deepgemm_head.py
```

### 005: SM120 activation and quantization fusion

This is a custom Triton implementation inspired by the [activation/requantization epilogue in Mega MoE](https://github.com/deepseek-ai/DeepGEMM/blob/b64107f2b9599ca76445b7f62eedf66bae1d095b/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh). It applies to grouped ReLU² experts whose down projection uses FP8 and whose up output is BF16. Other configured paths retain the original execution.

The quantizer computes ReLU² with BF16 rounding before calculating either block scale. Backward fuses the ReLU² derivative into the down-projection input-gradient epilogue, retaining its BF16 boundaries. This removes the post-activation BF16 write/read and the separate input-gradient intermediate write/read. It still saves the BF16 preactivation for backward, so this is not a claim of reduced saved-activation memory.

### 006: Muon implementation experiment

Batch FP32 momentum and Nesterov operations across parameters, then apply each Polar Express update batch with one mixed-dtype Triton kernel. Existing weight pointers are passed as a tuple, avoiding a device pointer-table allocation or a full FP32 copy of the BF16 updates. Reduced-precision momentum preparation retains the scalar path's rounding, and CPU/noncontiguous/reduced-precision parameter updates retain scalar application. The polar projection, parameter selection, learning rates and AdamW are unchanged. Checkpointed momentum values retain their existing meaning. The CUDA update uses fused multiply-add, so last-bit differences from other contraction choices remain possible.

The training entry point executes `optimizer.step()` eagerly and compiles its internal polar operations. This patch retains that arrangement. Capturing the entire optimizer step with `fullgraph=True` failed in the local PyTorch 2.12 check because of the tuple of weight pointers; full optimizer compilation is not part of this experiment.

### 007: row activity weighted Muon

This deliberately changes the update rule. After the polar projection, it weights output rows by the square root of their prepared update direction's RMS relative to the matrix's mean row RMS, clips those gains to `[0.5, 2]`, and restores the projected update's original Frobenius norm before casting back to the original update dtype. Final rounding makes norm preservation approximate. QKV blocks are treated independently, as before.

The hypothesis is that bounded row activity information can help convergence without changing total step magnitude. It relaxes orthogonality and adds reductions, so either convergence or speed may worsen. This is an experimental idea, not a claim of a new published optimizer or a proven improvement.

### 008: stride-aware RoPE

The CUDA path rotates Q/K in one Triton launch, reading their original strides and writing contiguous outputs. Backward applies the inverse rotation. It supports grouped query attention and retains explicit reduced-precision product rounding; cosine and sine remain non-trainable buffers. CPU execution retains the existing implementation.

Inductor may already remove some baseline copies or fuse pointwise operations differently. Only the full training result establishes whether this replacement helps.

### 009: producer epilogue fusion

Apply 005 first, then 009:

```bash
git apply --check experiments/005-sm120-fused-activation-quantization.patch
git apply experiments/005-sm120-fused-activation-quantization.patch
git apply --check experiments/009-sm120-producer-epilogue-fusion.patch
git apply experiments/009-sm120-producer-epilogue-fusion.patch
```

The up-projection GEMM computes a 128x128 output tile, rounds its preactivation to BF16, applies ReLU² with BF16 rounding, and directly generates both FP8 orientations and their independent block scales for the down projection. It still saves BF16 preactivations for backward. The new operation applies only to bias-free ReLU² experts with both projections in FP8; the other paths retain 005's execution.

This adapts the producer epilogue idea to SM120-compatible Triton operations. It does not execute the SM100-specific instructions used by Mega MoE. The fixed tile can increase register/shared-memory pressure or change GEMM accumulation order. The theoretical intermediate traffic removed is about 2.5 GiB per MoE forward/backward at batch 8, sequence 8192, top-k 2 and hidden width 2048, including 005's reductions; this is not a measured bandwidth or speed result.

Reverse in the opposite order before another independent experiment:

```bash
git apply --reverse experiments/009-sm120-producer-epilogue-fusion.patch
git apply --reverse experiments/005-sm120-fused-activation-quantization.patch
```

### 010: direct FP32 logits

Apply 004 first, then 010. Run `python experiments/check_deepgemm_head.py` before training. Reverse 010 before reversing 004.

This requests FP32 output from the head's forward GEMM. Under the current BF16 autocast configuration, both backward GEMM results still round to BF16 before their gradients return to FP32. The existing loss-side `.float()` then receives FP32 logits directly. At 65,536 tokens and vocabulary size 50,304, a BF16 logits matrix occupies about 6.14 GiB. If the baseline materializes the subsequent conversion, avoiding that intermediate removes about 12.28 GiB of logical write/read traffic per forward. Inductor may already fuse the conversion into the loss; in that case direct FP32 logits can increase memory and traffic. Peak allocation and traffic must be measured.

FP32 logits also change the values supplied to the loss and the upstream gradients received by the head. This is a numerical experiment as well as a memory/traffic experiment; it must be compared with 004 rather than credited solely to a backend change.

### 011: fused Polar Express update

This replaces the CUDA `a * X + B @ X` operation in each default Polar Express iteration with a Triton GEMM epilogue. It retains BF16 operands, FP32 accumulation and one BF16 output rounding. The Gram operations, iteration coefficients, momentum and parameter selection stay the same. CPU execution and the separate `split_baddbmm` path retain their existing implementations.

The baseline already uses `addmm`/`baddbmm`, which can fuse this epilogue internally. This experiment tests whether a specialized kernel helps at the actual optimizer shapes; it does not assume a missing baseline fusion. Four tile configurations are autotuned. A different reduction order can change last-bit results.

### 012: native scaled FP8 on SM120

This replaces each explicit FP8 partial product followed by FP32 block rescaling with [`tl.dot_scaled`](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html). Every existing 128-element scale is repeated across four hardware scale groups of 32 elements. It retains the same E4M3 values, FP32 accumulators, current scaling, 1x128 activation/gradient blocks and 128x128 weight blocks; it does not quantize again or adopt a coarser scale.

Offline SM120 compilation confirms native block-scaled E4M3 MMA instructions, rather than BF16 emulation. The patch changes the forward and input-gradient Triton GEMMs and also the Triton weight-gradient alternative. The configured DeepGEMM weight-gradient path stays in place. Four- and eight-warp variants give the autotuner alternatives with different register usage. Hardware accumulation can differ from the baseline's explicitly scaled partial sums, so numerical and convergence checks remain necessary.

Exercise both weight-gradient alternatives in the CUDA correctness check:

```bash
python experiments/check_grouped.py --triton-wgrad
```

### 013: native scaled producer fusion

Apply 005 and 009, then 013. This includes the native scaled GEMM changes from 012 in the fusion path, so do not apply 012 separately. Use `python experiments/check_grouped.py --triton-wgrad` to check the complete MLP and both weight-gradient alternatives. Reverse 013, then 009, then 005.

The up-projection producer and the remaining grouped GEMMs use native block-scaled FP8 instructions while preserving 009's saved BF16 preactivations and ReLU² rounding. Offline compilation confirms native E4M3 operations without BF16 emulation. Its default training producer still uses 255 registers per thread, so replacing software rescaling did not establish the hoped-for occupancy improvement. This isolates the arithmetic implementation from further launch tuning.

### 014: compact native scales

Apply 012, then 014. Use `python experiments/check_grouped.py --triton-wgrad` and reverse 014 before reversing 012. This is an alternative to the producer-fusion branch. Do not combine it directly with 004 or 013: those experiments add scale consumers and allocations that would also need adapting.

Quantizers still calculate the same FP32 power-of-two scale and the same FP8 values, but store its UE8M0 exponent byte. Native GEMMs consume that byte directly. Row, weight and ordinary column-scale storage shrinks from four bytes per scale to one; DeepGEMM's already-packed int32 column scales stay unchanged. CPU comparisons decoded the stored bytes back to the original scales exactly, including zero and tiny blocks. In one local input-gradient configuration, register allocation fell from 255 to 162 registers per thread; the training compiler and GPU timing may differ.

### 015: producer launch tuning

Apply 005, 009 and 013, then 015. Use `python experiments/check_grouped.py --triton-wgrad`. Reverse 015, 013, 009 and 005 in that order.

The producer autotunes between eight warps with two pipeline stages and sixteen warps with one stage. Its arithmetic and quantization stay the same. Local SM120 compilation uses 255 and 128 registers per thread respectively, with no spills. Both configurations consume roughly 64K registers per block, so the lower per-thread count does not itself imply more resident blocks. The experiment lets GPU timing decide between the different warp and pipeline arrangements.

### 016: compact scales in the fused branch

Apply 005, 009, 013 and 015, then 016. This incorporates 014's scale-storage idea throughout the fused branch; do not apply 014 separately. Use `python experiments/check_grouped.py --triton-wgrad`. Reverse 016, 015, 013, 009 and 005 in that order.

Starting from the unpatched baseline, the complete combination is:

```bash
git apply experiments/005-sm120-fused-activation-quantization.patch
git apply experiments/009-sm120-producer-epilogue-fusion.patch
git apply experiments/013-native-scaled-producer-fusion.patch
git apply experiments/015-native-producer-autotune.patch
git apply experiments/016-compact-scales-producer.patch
python experiments/check_grouped.py --triton-wgrad
```

The shared quantizers, producer allocations, scale stores, native GEMMs and saved backward tensors consistently use UE8M0 bytes for ordinary scales. DeepGEMM's packed column scales stay int32. FP8 values and decoded scale values match the earlier branch in the CPU checks. Both producer launch choices compile without spills, but register counts do not improve uniformly; their GPU timings determine whether this combination is useful.

### 017: more persistent FP8 workers

Apply 012 and 014, then 017. Use `python experiments/check_grouped.py --triton-wgrad`. Reverse 017, 014 and 012 in that order.

This changes one line: the grouped GEMMs launch twice as many persistent workers as there are SMs. The previous grid provides only one worker block per SM on average, so lower register use alone cannot fill the GPU with more blocks. The larger pool permits additional concurrent blocks when the selected tile's resource use allows it. Quantization and each output tile's arithmetic stay unchanged. More workers can also add overhead, so this is a scheduling experiment whose speed requires measurement.

### 018: more persistent BF16 workers

Apply 000, then 018. Use `python experiments/check_grouped.py` and reverse 018 before reversing 000. Compare with 000, rather than 001's DeepGEMM replacement.

This doubles the worker pool for the ordinary grouped kernel used by the BF16 control. It changes the same scheduling choice as 017 without introducing FP8 or a different GEMM library. CPU tile-coverage checks include every existing tile configuration and multiple worker counts; each output is written exactly once. All four BF16 configurations compile for SM120. Speed still requires a complete training comparison.

### 019–021: reduce input and weight preparation

These are independent follow-ups to 005. Apply 005 and one of 019, 020 or 021; reverse the selected patch before reversing 005. They are not a series to apply together.

019 gathers the original token activations directly in the up projection's quantizer, avoiding the intermediate packed FP32 input. It retains sorted expert order, the existing 128-element quantization blocks and both FP8 orientations. Backward accumulates the selected experts' contributions into each original token's gradient.

020 reads the individual expert parameters and writes grouped FP8 weights and scales in one quantization launch, avoiding the FP32 weight stack. The separate parameters retain their gradients, Muon updates and checkpoint names.

021 prepares detached FP8 expert weights and scales once per gradient-accumulation group and releases them before the optimizer update. Activations and their gradients are still prepared per microstep. The default model retains about 256 MiB of quantized weights and scales during accumulation; the original weight stack remains part of each forward. Accumulation count 1 and evaluation retain the uncached path.

To run the unpatched baseline, the 005 control and these three experiments:

```bash
python experiments/run.py 005 019 020 021
```

The runner applies and reverses 005 automatically for each dependent experiment. Local checks cover patch application/reversal, Python syntax and CPU-adapted quantization/autograd comparisons. CUDA execution, full-graph compilation and performance of 019–021 remain untested.

### 022: combine packing and direct weight quantization

Apply 005 and 019, then 022. This incorporates 020's direct expert-weight quantization into 019's packed-input operation; do not apply 020 separately. Compare with 005 + 019 to measure the additional weight-preparation change.

The up projection quantizes tokens directly in routed order and reads the individual expert parameters without stacking their FP32 weights. The down projection also quantizes individual weights and retains 005's fused ReLU² path. Backward accumulates each token's selected-expert contributions and returns gradients to the original expert parameters. The quantization blocks, scales, dtypes and configured weight-gradient backend remain unchanged.

Run the unpatched baseline followed by the combined experiment:

```bash
python experiments/run.py 022
```

The runner applies 005, 019 and 022, then reverses them in the opposite order. With the patches applied, `python experiments/check_grouped.py --triton-wgrad` checks the combined operation and both weight-gradient backends. GPU compilation, performance and convergence of this combination still require testing.

## Validation and interpretation

Each patch was applied and reversed against a clean baseline, and the resulting Python sources were parsed. CPU checks executed kernel bodies with tensor stand-ins or compared reference optimizer updates, including ragged and empty experts, noncontiguous tensors, low-precision boundaries, and tied gradients where applicable. These checks do not reproduce the GPU's accumulator behavior.

For 000–018, local offline compilation also generated SM120 cubins for the added Triton kernels and representative existing shapes/configurations. FakeTensor and full-graph capture checks exercised operator schemas and registered backward paths without GPU execution. These compiler checks used an existing local image with PyTorch 2.12 development and Triton 3.6; the intended training image uses PyTorch 2.16 development and Triton 3.9. They do not replace a CUDA correctness run on that image. The new DeepGEMM image has not been built, and no GPU timing or convergence result is claimed.

For 008, 012, 014, 016 and 017, scratch harnesses also reached Inductor's CUDA lowering/code generation and precompiled the Triton candidates into SM120 cubins. The FP8 checks used the actual batch-8 projection shapes and explicit backward operations, including fused ReLU² gradients. Device/driver setup was mocked and unrelated pattern initialization was disabled; generated kernels were not launched. This checks integration beyond isolated kernel compilation, rather than establishing a complete training run.

For each GPU run, record validation loss at matched steps, training step times at matched batch/window settings, and time to the validation target. The logged `total train time` excludes validation and initial warmup; include those separately when comparing total runtime. Start each independent training run with the same seed and data configuration rather than resuming one experiment from another's optimizer state.
