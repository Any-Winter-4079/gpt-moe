# Training experiments

The production code now includes the selected MoE, optimizer and attention changes. **Run `baseline` without applying the old winning patch combinations.** The default training configuration uses custom FP8 expert and attention projections, Transformer Engine FP8 for the LM head, and the normal precision context for the router. Validation, evaluation and sampling use `default` precision.

## Configuration

Each projection is configured in `TrainingConfig.linear_config[phase][role]` with `precision` (`default` or `fp8`) and `backend` (`custom` or `te`). The available implementations are:

| Operation | `default` + `custom` | `fp8` + `custom` | `fp8` + `te` |
|---|---|---|---|
| Grouped experts | Context precision, including the BF16 worker improvement | Fused routed preparation, cached weights, native scaled FP8 and DeepGEMM weight gradients | Not available |
| Ordinary MLP | Normal linear | Not available | Transformer Engine |
| Attention projections | Normal linear | Custom dense DeepGEMM projections with Q/K backward fusion where supported | Transformer Engine |
| LM head | Normal linear | Not available | Transformer Engine |
| Router | Normal linear | Not available | Not available |

Transformer Engine uses one E4M3 block-scaling recipe, with 1D activation/gradient scaling and 2D weight scaling. Unsupported precision/backend combinations raise a configuration error. `default` respects the configured context and tensor dtypes; it does not mean an unconditional BF16 cast.

## Run the consolidated baseline and follow-ups

Export `HF_USER` and `HF_TOKEN`, download the dataset, and run from the repository root using the DeepGEMM image described in the [main README](../README.md). These experiments do not require the FA4 image.

```bash
nohup python -u experiments/run.py baseline > baseline.log 2>&1 &
```

The runner accepts several names and executes them in the supplied order. Each experiment starts from the consolidated code, applies its own patch, trains, attempts to upload the results even after a training error, and reverses the patch before continuing. A repository lock prevents two runners from applying overlapping patches simultaneously.

| Name | Patch | Difference from the consolidated baseline |
|---|---|---|
| `baseline` | None | Current production configuration |
| `attention-te` | 049 | Transformer Engine for Q, K, V and output projections |
| `attention-qkv-only` | 033 | Keep custom FP8 Q/K/V; use context precision for the output projection |
| `attention-out-only` | 034 | Use context precision for Q/K/V; keep the custom FP8 output projection |
| `attention-bf16-dgrad` | 031 | Keep FP8 forward and weight gradients; calculate attention input gradients with BF16 matrix multiplication |

For example:

```bash
nohup python -u experiments/run.py baseline attention-qkv-only attention-out-only attention-bf16-dgrad > experiments.log 2>&1 &
```

No arguments runs only `baseline`. `--nproc-per-node N` selects the number of training GPUs. The previous adaptive and overnight queues belong to the historical runner.

The outer log prints the path of each training log. Follow that path with:

```bash
tail -n 50 -f experiment-baseline-YYYYMMDD_HHMMSS-01.log
```

Ctrl-C exits `tail`. On completion, the training log moves into `configs_and_logs/<timestamp>/console.log`; upload output and the batch summary remain at the repository root. To interrupt the runner, send SIGTERM to its PID and let it stop its training subprocess and reverse the patch. SIGKILL or terminating the instance cannot run that cleanup.

The four patches listed above have been ported to the consolidated file paths and configuration. Their GPU behavior and performance must be checked on the target machine. Patch 031 also adapts the fused Q/K backward, so the experiment changes all custom attention input-gradient paths rather than only the output projection. The unchanged CUDA reference checks describe the default production arithmetic; their FP8 input-gradient expectations do not apply to 031.

## Compare the previous winner with the integrated code

Use the **same GPU, image, power limit, dataset and training settings**, and run these comparisons sequentially. Compare the complete training runs, not only a kernel benchmark or the first step timings.

Parameter names and state-dict structure are preserved, but the trainer validates the saved configuration when resuming. Old checkpoints contain the previous configuration names and must be resumed from the old checkout; no automatic configuration migration is provided.

The old runner and all original patch versions are preserved at:

```text
00d0970a55f47999bc35b98d2e1a7ca3b91038da
```

After committing the consolidation, create a separate checkout of that reference. These commands share the already downloaded dataset without resetting or copying it. Run them from the consolidated repository root:

```bash
git worktree add --detach ../gpt-moe-reference 00d0970a55f47999bc35b98d2e1a7ca3b91038da
ln -s "$PWD/data/edu_fineweb10B" ../gpt-moe-reference/data/edu_fineweb10B
cd ../gpt-moe-reference
nohup python -u experiments/run.py 5+20+11+12+19+21+route+qkv-bq > reference.log 2>&1 &
```

Wait until that run and its upload finish, then return to the consolidated checkout:

```bash
cd ../gpt-moe
nohup python -u experiments/run.py baseline > consolidated.log 2>&1 &
```

The old combination is 027 + 042 + 029 + 030 + 044. Those are the previously winning custom FP8 expert/attention, optimizer and routing changes now integrated in production. The BF16 grouped worker improvement is also integrated, but it is exercised when expert projections use `default` under BF16 autocast.

For both results, record final validation loss, stopping step and logged **`total train time`** to the same validation target. This time excludes validation and initial warmup; record those separately if comparing wall-clock runtime. The run's `console.log` embeds the source code and the Hub upload preserves it. The [runner](run.py) records training/upload exits and timestamps; the [training entry point](../train.py) supplies the full-training measurement. Do not infer a gain by comparing absolute times from different rental machines.

## New candidates against the consolidated code

These are independent experiments against the current integrated baseline. They require the existing DeepGEMM image and add no dependencies. Each name applies only its listed patch; the runner restores the baseline between experiments. Commit the consolidation and these experiments together before pulling them onto the training machine.

| Name | Patch | Hypothesis | Numerical considerations |
|---|---|---|---|
| `polar-momentum-pack` | 064 | Fuse momentum, Nesterov and direct BF16 packing, avoiding per-parameter pointwise launches and the FP32 gradient stack | Keeps separate momentum multiply/add rounding; Nesterov FMA and device results need GPU comparison |
| `polar-final-update` | 055 | Fuse the last Polar Express GEMM with FP32 parameter updates, avoiding the final BF16 update buffer and separate parameter-update launches | Keeps momentum, coefficients and the BF16 update boundary; fixed GEMM tiles can change reduction rounding |
| `qk-token-major-layout` | 063 | Store normalized Q/K in token-major storage so attention can produce an output that needs no transpose copy | Changes strides, not Q/K arithmetic; the attention kernel/layout choice can change |
| `attention-output-layout` | 062 | Quantize directly from attention's B,H,T,D output instead of materializing a reordered BF16 matrix | Preserves quantization blocks; retains the existing route when the transpose is already contiguous |
| `rmsnorm-fp8-preparation` | 058 | Normalize directly while producing both FP8 attention-input layouts, avoiding an intermediate normalized FP32 matrix | Preserves RMSNorm epsilon, learned-weight gradients and the existing fused QKV backward; reduction order can differ |
| `qkv-head-tiled-backward` | 054 | Reduce live Q/K derivative intermediates by processing one head at a time before quantization | Retains the shared 128-feature row scales, 128-token column scales and BF16 boundaries |
| `moe-counted-route` | 052 | Replace general sorting and routing scatters with histograms, prefix sums and bucket permutation | Stable within-expert ordering can differ from an unspecified-stability sort, affecting FP8 scale groups |
| `moe-output-combine` | 053 | Fuse inverse gather, gate weighting and top-k reduction, with a specialized backward that saves the original packed tensor | Preserves cast/product boundaries; router-weight reduction order can differ |
| `streaming-cross-entropy` | 059 | Compute FP32 loss statistics directly from BF16 logits without full FP32 logits/log-softmax buffers | Keeps the TE FP8 head and FP32 loss; online reductions can differ slightly |
| `integer-fp8-scales` | 060 | Replace log2/ceil/exp2 scale construction with integer exponent/mantissa operations | Same power-of-two rule; can round upward where approximate log2 misses a near-boundary FP32 value |
| `residual-rmsnorm` | 057 | Fuse the learned attention residual with the following RMSNorm and both backward paths | Preserves learned residual/norm gradients; reduction order can differ |
| `polar-gram-trace` | 056 | Reuse the first Gram matrix to derive normalization instead of scanning the gradient separately | **Changes optimizer numerics substantially; test separately** |
| `lm-head-custom` | 061 | Try the existing custom dense DeepGEMM projections for the LM head, with int64 quantizer indexing | Changes the FP8 head backend and promotes shared dense quantizer addresses; tied weights and cross-entropy stay in place |

All candidates passed independent application, Python syntax and byte-exact reversal checks, plus CPU algebra or indexing checks appropriate to the change. Independent source reviews covered their backward paths and indexing. **GPU compilation, correctness, speed and convergence remain untested.** CPU emulation does not establish CUDA correctness or a performance gain. In particular, test the batch-size transition from 4 to 8 before trusting a full run.

The suggested efficiency queue starts with one explicitly requested baseline on the same machine:

```bash
nohup python -u experiments/run.py baseline polar-momentum-pack polar-final-update qk-token-major-layout attention-output-layout rmsnorm-fp8-preparation qkv-head-tiled-backward moe-counted-route moe-output-combine streaming-cross-entropy integer-fp8-scales residual-rmsnorm > new-candidates.log 2>&1 &
```

If that machine already has a comparable consolidated baseline, omit `baseline`. No arguments still runs only the baseline; no extra comparison run is inserted automatically. Compare **total training time to the same validation target**, final validation loss and stopping step. These candidates are all separate baseline comparisons, not a cumulative combination. Prepare and test a combined candidate only after identifying improvements.

The two attention-output layout experiments address the same copy at different points. Try them separately: 063 changes upstream Q/K storage, while 062 consumes the existing layout during FP8 preparation. Both retain FlexAttention and SDPA support. Current upstream [FlexAttention derives output strides from the query](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/flex/flex_attention.py#L424-L435); the installed version must still be checked. The token-major Q/K views are noncontiguous, so the current activation-offloading path would keep those tensors on the GPU. The default configuration has offloading disabled.

The Gram-trace experiment is deliberately outside that queue:

```bash
nohup python -u experiments/run.py polar-gram-trace > polar-gram-trace.log 2>&1 &
```

It follows the initialization idea in [modded-nanogpt's ANVIL](https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/optim/anvil.py), while keeping this project's existing momentum and five polynomial coefficients. Forming the first Gram matrix before normalization changes BF16 rounding: random CPU cases showed up to 2.57% relative update difference, and one rank-one case reached 83%. This is a convergence experiment, not an equivalent implementation with only lower overhead. The final-update candidate is separate from the previously rejected foreach and triangular optimizer variants.

The other new optimizer candidate, momentum preparation/packing, is also distinct from foreach: it writes the BF16 Polar Express input directly while updating momentum, rather than constructing an intermediate FP32 stack. The final-update and preparation candidates operate at opposite ends of the optimizer. Their individual results should be established before preparing a combined patch. Applying 055 and 064 mechanically would bypass 055 in the new preparation bucket loop; a combined candidate must adapt that loop explicitly.

The custom LM-head candidate is also outside the initial queue. It fixes a concrete indexing risk in the shared dense quantizers: a batch-8 gradient contains 65,536 × 50,304 = 3,296,722,944 elements, beyond signed int32. That is a plausible explanation for the old custom head's batch-transition failure, not a confirmed diagnosis. The new candidate reuses the current dense DeepGEMM forward and two backward GEMMs with FP32 parameter gradients. It includes `experiments/check_head.py` when applied. For a manual correctness check, with no experiment runner active:

```bash
git apply --check experiments/061-safe-dense-fp8-lm-head.patch
git apply experiments/061-safe-dense-fp8-lm-head.patch
python experiments/check_head.py
python experiments/check_head.py --large-offsets
git apply --reverse experiments/061-safe-dense-fp8-lm-head.patch
```

The large-offset check needs about 14 GB of free GPU memory and exercises both quantized layouts at the real batch-8 gradient shape. It does not test the full-size GEMM or complete training. After reversing the patch, run the isolated experiment normally:

```bash
nohup python -u experiments/run.py lm-head-custom > lm-head-custom.log 2>&1 &
```

Research informing the other candidates includes [Megatron's expert permutation](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/inference/moe/permute.py), [MegaBlocks' weighted scatter/backward](https://github.com/databricks/megablocks/blob/main/megablocks/ops/scatter.py), [FlashAttention residual normalization](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py), [FlashAttention cross-entropy](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/cross_entropy.py), and [Triton's register/occupancy discussion](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html). These patches adapt the ideas to the existing model; they do not install or switch to those projects.

Performance questions to resolve on the GPU:

- Inductor may already fuse much of the ordinary residual or weighted-combine forward expressions. Explicit kernels can also prevent surrounding fusion; saving backward state is part of the combine hypothesis.
- The normalization candidates use FP32 partial weight-gradient buffers: 16 MiB at 65,536 tokens and width 1,024. That traffic may offset their forward savings.
- The counted router targets the current eight-expert setup. Its one-CTA prefix stage and per-expert scans can become expensive with many more experts.
- Streaming cross-entropy trades large workspaces for serial vocabulary traversal and more backward blocks. Both kernels use int64 row offsets because 65,536 × 50,304 exceeds 2³¹ elements.
- The integer-scale candidate matched the CPU reference for every finite nonnegative BF16 maximum and more than four million random/boundary FP32 cases. Approximate GPU division/log instructions still need comparison on the pinned build.

## Historical patches

Numbered patches 000–051, except for the four ported patches above, remain historical experiments. They do not target the consolidated code. The original production baseline was commit `40af18c6ded4916a59b7e145ada749e459370218`; dependencies and combined candidates were registered in the runner at `00d0970a55f47999bc35b98d2e1a7ca3b91038da`. Check out that reference to reproduce them, including the original versions of 031, 033, 034 and 049.

The current runner rejects old IDs and combinations to prevent applying already integrated code again. Other historical candidates, including MLA, exact sliding windows, FA4, CUDA projection kernels and alternate FP8 kernels, require adaptation before they can be tested against the consolidated baseline.

The integrated changes correspond to:

| Historical change | Production behavior |
|---|---|
| 005 | ReLU²/quantization and derivative fusion; other activations retain their normal path |
| 020 | Quantize expert parameters directly without an FP32 weight stack |
| 019 | Gather routed tokens while quantizing activations |
| 021 | Reuse quantized expert weights across accumulation microsteps; clear before updates and on phase/load changes |
| 012 | Native scaled FP8 grouped GEMMs; DeepGEMM weight gradients |
| 011 | Fused Polar Express matrix product and scaled residual |
| 042 | Route input gradients back without atomic accumulation |
| 039 | Fused Q/K normalization, RoPE and learned scaling on the normal attention path |
| 029 + 030 + 044 | Custom FP8 Q/K/V/output projections and fused Q/K backward quantization |
| 018 | Larger persistent worker pool for ordinary grouped BF16 projections |

Historical patch names are retained for traceability; they do not prescribe an execution order or a queue of untested experiments.
