# Exact static Flex masks for evaluation

Baseline: `7704a680a84be3000e407887e149914215c655c5`.
Patch: `06-static-validation-mask.patch`, changing only `model/gpt.py`.
Status: candidate for a separately approved GPU experiment; CPU mask equivalence verified.

The supplied log spends about 49.38 minutes in validation across 73 events, compared with 53.44 minutes of training and 28.30 minutes of compilation. This experiment targets one plausible validation cost: rebuilding a token-level Flex mask on every evaluation batch. The log alone does not identify how much time this operation consumes.

The patch adds 26 net lines to `_build_flex_attn_block_mask`. Evaluation calls with no padding mask and no active document mask classify tiles from their minimum and maximum token distances. This produces the exact existing causal/noncausal, local/global mask without evaluating every token pair. It uses the existing `BlockMask.from_kv_blocks` interface and existing mask function, with ascending valid tile ordering matching PyTorch's baseline builder. Padding and document-dependent evaluation masks use the existing generic path.

For tile-center distance `d = (query_tile - key_tile) * block_size`, the nearest absolute token distance is `max(abs(d) - block_size + 1, 0)` and the farthest is `abs(d) + block_size - 1`. The first determines whether SWA permits any pair; the second determines whether it permits every pair. Causality independently requires `d >= 0` for any pair and `d >= block_size - 1` for every pair. Full tiles can therefore safely skip the mask function, and partial tiles use precisely the existing per-token predicate.

The gate adds no configuration flag or alternate attention backend. It retains the current SWA tensor as a dynamic input, with no cached window value. Training, evaluation loss reduction, token order, model state, and full-layer routing are unchanged. The change is shared by the existing DDP and PP model calls; distributed execution itself was not tested locally.

## Checks

Local CPU PyTorch 2.9.0; the two mask-builder methods were extracted directly from the baseline and candidate ASTs. The reference `create_block_mask` was forced onto CPU and its compilation wrapper disabled. These checks execute mask operations only, with no attention, model forward, training, inference job, or remote execution.

- 672 static cases: block sizes 1, 2, 4, 16, 64, and 128; one, two, or four blocks per sequence; batches 1 and 3; causal/noncausal; SWA/global; windows at negative, zero, one-token, tile boundaries, nonmultiples, sequence boundaries, and beyond the sequence. All eight partial/full KV/Q metadata tensors, shape, and block size match `create_block_mask` exactly. Reconstructing effective token masks from full and partial metadata also matches an independent token-level predicate.
- 192 fallback/training cases: combinations of training/eval, causal/noncausal, SWA/global, document masking enabled/disabled/ignored, absent/monotone/unordered document IDs, and random padding. Baseline and candidate metadata match in every case. All evaluation cases also match the independent token predicate. A call counter confirms generic-mask fallback selection.
- 20 normalization-contract cases: 2D masks and invalid shapes, including 3D and 4D, through both Flex and SDPA normalization. The baseline accepts only 2D masks; its values and error messages are preserved exactly.
- Full-graph Dynamo capture of mask construction with an eager backend produced one graph across six in-place SWA updates: 4, 5, 8, 1, 32, and 0. Every result matches the reference. This checks frontend capture and dynamic-window handling, not CUDA/Inductor compilation.
- Candidate Python syntax parses; `git apply --check` against the baseline passes.

CPU-only eager mask-construction medians, one thread, one warmup and five timed calls, window 384 and block size 128:

| Batch | Sequence length | Baseline | Candidate |
| --- | --- | --- | --- |
| 1 | 1024 | 2.441 ms | 0.234 ms |
| 3 | 2048 | 20.121 ms | 0.264 ms |
| 1 | 8192 | 195.517 ms | 0.393 ms |

These timings cover only CPU mask construction. They are not GPU or end-to-end speedup estimates. The compiled GPU baseline may fuse temporary tensors; the benefit must be measured in the target CUDA configuration. Logits, loss, transfers, and attention may account for much of validation time.

The local check source is `/tmp/gpt-moe-validation-static/check.py`, with output `/tmp/gpt-moe-validation-static/check.log`. It can be rerun while those scratch files exist with:

```sh
/var/folders/5j/rxlcfqfs01d82ch5y38n7rt80000gn/T/gpt-moe-opcheck-ntzu20_x/bin/python /tmp/gpt-moe-validation-static/check.py
```

## Existing semantic difference and experiment boundary

Do not replace `if self.is_causal and self.training` with `if self.is_causal`. The existing training branch approximates the SWA boundary using block counts. At sequence length 256, block size 128, window 128, batch 1, and no document/padding mask, its effective mask omits 8,256 pairs that the baseline evaluation predicate permits. This pre-existing difference is preserved; the candidate matches evaluation, not the approximate training mask.

Before adoption, a separately approved run should compare validation losses and wall time on identical checkpoints, token batches, windows, and compilation settings. GPU/Inductor lowering, CUDA graph behavior, and actual DDP/PP execution remain unverified. No GPU speedup is claimed by this patch.
