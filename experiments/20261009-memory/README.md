# Memory experiments — 2026-10-09

Three independent patches against the working tree at `f06e05b46c7202bb5f91859b0095ba53aab0292f`, **including the pending whole-block activation-checkpointing changes** in `config/training.py` and `model/gpt.py`. Commit those existing changes with this directory before pulling on the training machine. The patches are not applied to the main code.

These are hypotheses to measure, not established improvements. Test one patch at a time. None changes the model dimensions, attention pattern, token budget, optimizer arithmetic, or data order. Current baseline configuration has `use_activation_checkpointing = True`.

| Patch | Question | Main tradeoff |
| --- | --- | --- |
| `01-mlp-checkpointing.patch` | Can recomputing just the MLP/MoE recover some speed while retaining useful memory savings? | Retains attention intermediates; should save less memory than whole-block checkpointing |
| `02-muon-momentum-offload.patch` | Is moving Muon momentum once per optimizer update practical? | Pinned host RAM and CPU↔GPU transfers; arithmetic remains on the GPU |
| `03-pipeline-split.patch` | Can an unequal two-stage split reduce the larger stage's memory peak? | Moving layers can create a compute imbalance and reduce throughput |

## Apply and reverse

From the repository root, with no other experimental patch applied:

```sh
git apply --check experiments/20261009-memory/01-mlp-checkpointing.patch
git apply experiments/20261009-memory/01-mlp-checkpointing.patch
```

Reverse after testing:

```sh
git apply -R --check experiments/20261009-memory/01-mlp-checkpointing.patch
git apply -R experiments/20261009-memory/01-mlp-checkpointing.patch
```

Substitute the filename for experiments 02 and 03. Undo your experiment-specific flag edits before reversing if they overlap a patch hunk. A failed `--check` means the local source differs from this baseline; do not force the patch.

## 01 — MLP/MoE checkpointing

Changes `config/training.py`, `model/gpt.py`, and `model/block.py`.

The patch adds `activation_checkpointing_scope = "mlp"`. It checkpoints the MLP or MoE call, including routing and its balance loss, in both residual-path variants. The attention and normalization calls remain outside that checkpoint. Evaluation bypasses checkpointing. It uses the same non-reentrant PyTorch checkpoint API as the baseline.

Compare three settings at fixed model size, sequence length, batch size, and attention:

- `use_activation_checkpointing = False`: no explicit activation checkpointing.
- `use_activation_checkpointing = True`, scope `"block"`: current whole-block implementation.
- `use_activation_checkpointing = True`, scope `"mlp"`: this experiment.

The question is the memory/time tradeoff, not whether recomputation makes the same workload intrinsically faster. The common model path supports DDP and PP, but compiled execution with this project's custom MoE backward still needs a GPU test. Start with the current single-GPU configuration.

## 02 — Muon momentum offload

Changes `config/training.py`, `model/gpt.py`, and `optimizers/muon.py`.

The patch enables `muon_offload_momentum = True`. Set it to `False` for the resident baseline. Parameters, gradients, AdamW states, and orthogonalization remain on the GPU. Only each Muon momentum buffer resides in pinned CPU memory between optimizer updates.

For each parameter with a gradient, the buffer is copied to the GPU, updated using the existing operations, and copied back. The return copy is blocking so it is finished before the host buffer can be reused or saved. This simple experiment does **not** overlap transfers on a separate CUDA stream. It should not be described as an optimized offload implementation.

For `P` Muon parameters with FP32 momentum, persistent GPU state removed is approximately `4P` bytes, replaced by roughly the same amount of pinned host memory. Transfer volume is approximately `8P` bytes per optimizer step when every parameter has a gradient. For 2.718 billion Muon parameters, that is about 10.87 GB of momentum and 21.75 GB transferred per update. Read the actual Muon parameter count in the startup log; peak GPU memory savings need not equal the persistent-state reduction. Allocator-reserved memory may also remain high.

The transfer happens once per optimizer update, rather than once per microbatch's forward/backward. With 16 accumulation steps this is a materially different test from the rejected activation-offloading experiment. It may nevertheless be slower, especially with many small transfers.

Use a fresh run for the first benchmark. PyTorch optimizer loading can temporarily copy saved momentum back onto the GPU; this patch returns it to pinned CPU storage at the next update, but does not eliminate that loading peak. A parameter without a gradient is skipped as before. The offload flag is configured when constructing the optimizer, not changed halfway through a run.

Before a full training comparison, this optional check runs small synthetic optimizer updates on the GPU, including the Polar Express batch path, QKV splitting, missing gradients, both Nesterov settings, and a state-dictionary reload:

```sh
python experiments/20261009-memory/check_muon_offload.py
```

Run it only after applying patch 02, in the normal CUDA training environment. It compiles the optimizer kernels as needed. This check has not been run here, and passing it does not establish full-model throughput or memory savings.

## 03 — Unequal two-stage pipeline split

Changes `config/training.py`, `model/gpt.py`, `train.py`, and `checkpointing/checkpoint.py`.

`pipeline_split_layer = -1` preserves the existing halfway split. A positive value sets the number of transformer blocks on stage 0; the rest go to stage 1. Both stages must retain at least one block. This remains the existing two-stage 1F1B pipeline schedule.

For the 18-layer model, compare `-1` (9/9) with `8` (8/10) if stage 0 has the larger peak. If stage 1 is the limiting stage, try `10` (10/8) instead. Embedding/output-layer ownership and the original per-layer attention assignments remain intact.

This test needs two GPUs, `parallel_mode = "pp"`, and `use_tied_embeddings = False` in the model configuration. Those settings must match in the baseline and candidate; the patch does not switch the current DDP run into PP. Match tokens per optimizer update and per-microbatch shape across comparisons.

Explicit split settings are recorded in resume configuration. Full-model export uses that same boundary when renumbering stage 1's blocks. Existing default-split checkpoint exports use the halfway fallback. Resume expects the same split configuration; this patch does not repartition saved optimizer state.

## Measurements and local checks

First record the existing whole-block checkpointing result, then try 01, then 02 independently; save 03 for a two-GPU session. Keep hardware, attention, precision, token counts, and other settings fixed. Compare token loss as well as throughput so a faster but numerically broken run is not accepted.

Record compilation/warmup separately, steady training step time, tokens/s, per-GPU peak allocated/reserved memory, and peak host RAM. For PP, retain the numbers for each stage rather than only their sum. Validation and checkpoint saving are separate phases with potentially different peaks. Use the same measurement method in every comparison.

Local checks passed: Python syntax for all candidate files; each patch applies to the current working tree and reverses to identical source bytes; the actual PP partition/export logic was exercised with stand-in weights across 10 valid layouts and 12 invalid split cases. This verifies layer placement and export naming, not distributed execution. PyTorch/CUDA are unavailable in the local environment, so numerical equivalence, custom-kernel compilation, GPU memory, and performance remain untested.

PyTorch references: [checkpoint API](https://docs.pytorch.org/docs/stable/checkpoint.html) and [pinned memory and nonblocking copies](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
