# Fused Lion optimizer

Independent patch against the working tree of 2026-10-09, including its existing whole-block activation checkpoint edits. Applying this patch selects `optimizer_type="lion"`. Select `"muon"` or `"adamw"` to use the existing optimizer paths.

The new `optimizers/lion.py` implements the [official Google Lion equations](https://github.com/google/automl/blob/master/lion/lion_pytorch.py): decoupled weight decay, `u = beta1 * m + (1 - beta1) * g`, a signed parameter update, then `m = beta2 * m + (1 - beta2) * g`. One Triton launch per parameter fuses these elementwise operations and writes one FP32 momentum buffer. This is per-tensor fusion, not a single launch covering all model parameters. No optimizer implementation is installed or downloaded at runtime; Triton is already used by this repository.

| Configuration | Patch default | Meaning |
| --- | --- | --- |
| `optimizer_type` | `"lion"` | Lion for all trainable parameters |
| `lion_betas` | `(0.9, 0.99)` | Update interpolation and momentum EMA |
| `lion_lr_scale` | `0.1` | Multiplier on the existing token-based base LR schedule |
| `lion_weight_decay` | `0.1` | Decoupled decay on tensors with at least two dimensions |

The base schedule still uses the existing `adamw_max_lr`, warmup/cosine fields and `adamw_hard_min_lr`. The patch default makes Lion start at `0.0005` when the base rate is `0.005`. The multiplier also applies to the schedule floor. The rates and decay are starting experiment settings, not tuned results; sign updates change the optimization problem and require a fair convergence comparison.

Embeddings, head and matrix weights use Lion with decay; scalar/vector parameters use Lion without decay. `named_parameters()` deduplicates tied weights. Each DDP rank updates its own already-reduced gradients; a PP process updates only parameters present in its local stage. The patch adds no collective or parameter sharding. Missing gradients skip both decay and momentum creation/update; a present zero gradient still updates according to Lion. Dense CUDA FP32, BF16 and FP16 parameters must be contiguous; noncontiguous gradients are copied before the kernel. Sparse gradients are rejected.

Arithmetic and momentum stay FP32. Low-precision parameters are rounded once when the kernel stores the complete update; there is no FP32 master-weight copy. Consequently, this is not bitwise equivalent to running the upstream eager optimizer directly on BF16/FP16 weights, whose intermediate operations round in parameter dtype. Small updates may still round away at the final parameter store. `load_state_dict` restores the original saved FP32 momentum values after PyTorch's default state casting, including the generic warmup snapshot/restore path. The three new options are included in checkpoint configuration matching. Start a fresh optimizer experiment: old baseline checkpoints lack those options and the optimizer name/state is different.

A faster optimizer step does not establish faster training to a target loss. Measure steady-state optimizer time and full-step time separately, then compare held-out loss at equal tokens and equal wall time after tuning. Kernel-launch overhead may dominate small tensors, and a noncontiguous-gradient copy costs additional memory traffic. This patch makes no speed or quality claim.

Local validation completed: syntax parsing of all patched Python sources and the checker; forward applicability against the live working tree; whitespace checks; isolated apply/reverse restoring the copied baseline byte-for-byte; extracted-method checks with parameter/optimizer stubs for full-model and both PP-stage ownership, disjoint parameter coverage and LR scaling. These stub checks do not execute PyTorch or a distributed backend. PyTorch, Triton and CUDA execution were unavailable locally. The optional tensor-only checker covers FP32-equation parity across dtypes, non-block-aligned sizes, zero and missing gradients, FP32 momentum reload, continued updates and warmup reset. It has not been executed.

Apply independently from the repository root:

```sh
git apply --check experiments/20261009-additional/08-fused-lion.patch
git apply experiments/20261009-additional/08-fused-lion.patch
```

Optional CUDA algebra checks, only when explicitly requested:

```sh
python experiments/20261009-additional/check_lion.py
```

Reverse after the experiment, before applying another independent patch:

```sh
git apply --reverse --check experiments/20261009-additional/08-fused-lion.patch
git apply --reverse experiments/20261009-additional/08-fused-lion.patch
```
