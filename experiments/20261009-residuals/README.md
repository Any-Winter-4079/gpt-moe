# Five independent residual-path experiments — 2026-10-09

These patches test changes to model architecture or initialization. The objective is less wall-clock time to a target validation loss. They may increase training step time; none has a measured speedup or convergence benefit yet. They are separate from the ten implementation-only experiments in `20261009-throughput`.

Each patch applies independently to the current working tree at `f06e05b46c7202bb5f91859b0095ba53aab0292f`, including the pending whole-block activation-checkpointing changes. Production files remain unchanged. Apply one patch at a time. Every patch adds configuration in `config/model.py` and enables its experiment by default; set its new boolean to `False` before starting a run to recover the original equations and initialization.

| Patch | Configuration | Default experiment |
| --- | --- | --- |
| `01-gated-long-skips.patch` | `use_long_skips`, `long_skip_span`, `long_skip_init` | A learned skip around each complete group of two blocks; scalar gate starts at zero |
| `02-token-embedding-reinjection.patch` | `use_embedding_reinjection`, `embedding_reinjection_init` | Add a gated copy of the original token embeddings before each block after block 0; gates start at zero |
| `03-parallel-attention-mlp.patch` | `use_parallel_attn_mlp` | Attention and MLP each normalize the same block input |
| `04-channelwise-path-scales.patch` | `use_channelwise_path_scales` | Existing enabled main/residual scales become vectors of length `d_model` |
| `05-depth-scaled-main-init.patch` | `use_depth_scaled_main_init`, `depth_main_init`, `depth_main_power` | Initialize learned main-path scales to `1 / sqrt(2 * (layer_index + 1))` |

## Apply and reverse

From the repository root:

```sh
git apply --check experiments/20261009-residuals/01-gated-long-skips.patch
git apply experiments/20261009-residuals/01-gated-long-skips.patch
```

To reverse:

```sh
git apply -R --check experiments/20261009-residuals/01-gated-long-skips.patch
git apply -R experiments/20261009-residuals/01-gated-long-skips.patch
```

Substitute another filename for a different experiment. Restore any manual edits to the patch's configuration lines before reversing it. Use a fresh run for each comparison: model configuration is already included in strict resume checks, and some experiments add parameters or change parameter shapes. These patches do not migrate old checkpoints. They overlap in source locations, so do not stack them for the initial tests.

## What changes

### 01 — Gated skips across blocks

For each complete group, retain its input `s`, execute the existing blocks, then add `gate * s` to the group output. With the default span of two, the skips surround blocks `[0, 1]`, `[2, 3]`, and so on. An incomplete final group has no added skip. The span can be any integer from two through the model's layer count.

Only the last block in each complete group owns a gate. Zero initialization preserves the initial forward function, while the gate can receive gradients immediately. The existing short residual paths stay in place. The 18-layer configuration adds nine scalar parameters at span two. Saved skip inputs can extend activation lifetimes, so measure peak VRAM with checkpointing both considered and held constant across comparisons.

Groups use global layer indices. If a group crosses the existing two-stage pipeline split, its input travels as an extra tensor alongside the hidden state and MoE balance loss. Its gradient travels back through the scheduler. Groups that end exactly at the split need no extra transfer. For the current 18-layer model, span two crosses the split after block 8.

### 02 — Original token-embedding reinjection

Before block `i > 0`, compute `x = x + embedding_scale_i * token_embeddings`. The reference is the original token-embedding output; absolute positional embeddings, when enabled, are added normally at model entry and are not repeatedly injected. RoPE/NoPE behavior is unchanged.

Each receiving block owns its gate, adding 17 scalar parameters to the current 18-layer model. The gates start at zero. Gradients also flow to the original embeddings; the reference is not detached. This keeps that tensor live across depth. Pipeline stage 0 sends it to stage 1 as an extra tensor, with its own shape and dtype, for training, validation and sampling.

For both 01 and 02, a `[1, 16384, 2048]` reference is 128 MiB in FP32 or 64 MiB in BF16. That is the size of the extra reference tensor and, when transported, its forward payload; there is also backward communication. It is **not** an estimate of the total additional peak memory, which depends on aliases, saved tensors and the pipeline schedule. Both patches keep new parameters under their owning blocks so existing stage checkpoint/export key mapping still applies. A zero balance-loss placeholder allows the same three-tensor pipeline handoff in dense models; it is excluded from the dense model's loss calculation.

### 03 — Parallel attention/MLP inputs

With the current main-path gates, the new block equation is:

```text
x_out = x + beta_attn * Attention(norm1(x)) + beta_mlp * MLP(norm2(x))
```

The MLP no longer consumes the post-attention residual. Both normalizations, attention masks, MoE routing/balance loss and existing gate options are retained. If learned residual-path scales are also enabled, their existing application order remains: `alpha_mlp * (alpha_attn * x + beta_attn * A(x)) + beta_mlp * M(x)`. The unweighted branch retains its existing fixed scale.

This removes a dependency in the architecture. It does not add CUDA streams or guarantee concurrent kernel execution; the attention and MLP calls still appear sequentially in Python. A lower step time or fewer steps must be demonstrated experimentally.

### 04 — Per-channel path scales

The existing enabled scale parameters change from shape `[1]` to `[d_model]`. Their initialization values and forward equations remain the same, now broadcasting one learned coefficient per channel. This applies to whichever of the main/residual paths are already enabled; it does not turn either path on. With both paths disabled, it has no effect and leaves the fixed scale alone.

For the current configuration, only main-path scales are enabled: 36 scalars become 36 vectors of 2,048 values, adding 73,692 parameters. They remain one-dimensional parameters handled by the existing non-decayed AdamW group. This offers more flexibility, with additional scale/optimizer work; it is not a direct arithmetic reduction.

### 05 — Depth-dependent main-path initialization

The formula is `depth_main_init / (2 * (layer_index + 1)) ** depth_main_power`. Defaults are `1.0` and `0.5`, giving approximately 0.707 in block 0 and 0.167 in block 17. Both attention and MLP main scales use this initialization, then train normally. Global indices preserve the same initialization in both pipeline stages. This option requires `use_weighted_main_path=True`.

The current baseline initializes those scales to zero. Multiplying that zero by a depth factor would do nothing, so this experiment has a separate nonzero starting value. Disabling the flag uses the original `weighted_main_path_init` again. Setting the power to zero provides a constant nonzero initialization comparison, useful for separating the effect of nonzero initialization from the depth taper. This patch adds no forward operations beyond the already-existing learned scale multiplications.

## Verification and measurement

All five patches passed independent application, Python syntax, whitespace and exact-reversal checks. Existing production source hashes are unchanged. Local algebra/control-flow checks executed the extracted source with NumPy stand-ins: 2,880 GPT routing cases, 96 block-equation cases and 144 initialization cases. These checked zero-gate/disabled equivalence, whole-model versus two-stage composition, crossing/aligned skips, incomplete groups, dense/MoE balance handling, absolute/token embedding separation, and weighting combinations. The checkpoint stand-in simply calls the block: these checks do not exercise PyTorch autograd, actual checkpoint recomputation or CUDA.

An optional small CUDA component check is included for later use after applying one patch:

```sh
python experiments/20261009-residuals/check_residuals.py
python experiments/20261009-residuals/check_residuals.py --bf16 --compile
```

It compares whole-model and two-stage composition on **one GPU**, including logits, loss and parameter gradients, with checkpointing on/off, dense/MoE blocks and RoPE/absolute positions. It uses nonzero skip gates to exercise their gradients. The compile option additionally compares compiled/eager dense forward and backward. These commands have not been run here because local PyTorch/CUDA are unavailable. This is not a distributed pipeline test: actual DDP/1F1B transport, FlexAttention, full-size compilation and throughput remain to be tested on the training machine. PyTorch's [pipeline documentation](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html) describes the fixed shape/dtype requirements for stage communication.

Keep model dimensions, attention policy, token budget, checkpointing and hardware fixed. Record steady step time, tokens/s, peak VRAM and validation loss at equal consumed tokens. Because these change the model or its initialization, also compare total time to the same validation loss; faster individual steps alone do not establish a better experiment.
