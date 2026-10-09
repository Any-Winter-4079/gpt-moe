# 05 — Gram Newton-Schulz backend for the existing Muon

Independent patch: `05-gram-newton-schulz.patch`. Apply to the current baseline, including the pending whole-block checkpointing edits. The patch selects the new backend by default and retains the existing local Muon optimizer, its momentum state, Nesterov update, groups of at most eight matrices, shape-based three-way QKV split, and `sqrt(max(rows, columns))` update scaling. AdamW ownership and the learning-rate schedule stay the same.

This uses the public `GramNewtonSchulz` callable from [Dao-AILab at revision e45d0aca7083cb275c9a303220c05c4abecd9187](https://github.com/Dao-AILab/gram-newton-schulz/blob/e45d0aca7083cb275c9a303220c05c4abecd9187/gram_newton_schulz/gram_newton_schulz.py), pinned in `requirements.txt`. Rectangular matrices use Gram iteration; square matrices use the library's standard Newton-Schulz path. The hypothesis is less rectangular-matrix multiplication during orthogonalization; actual step time is unmeasured.

## Controls

| TrainingConfig field | Patch default | Meaning |
| --- | --- | --- |
| `muon_backend` | `"gram_newton_schulz"` | Select this experiment; `"polarexpress"` selects the original backend |
| `muon_backend_steps` | `5` | Fixed length of the existing coefficient sequence |
| `muon_gram_use_kernels` | `False` | PyTorch matrix operations; `True` enables upstream CuTeDSL kernels where supported |
| `muon_gram_reset_iterations` | `[2]` | Recompute the Gram matrix after two iterations; e.g. `[1, 2, 3, 4]` restarts after every non-final iteration |

Both new settings are included in the strict resume configuration. Use a fresh run for the comparison; old checkpoints have a different configuration schema. Restart indices refer to completed iterations and meaningful entries are 1 through 4. More restarts change rounding and trade additional matrix work for stability.

## Coefficients and numerical changes

The repository's five Polar Express coefficient triples are used instead of the dependency's different preset. The repository normalizes the BF16 input by `1.02 * norm + 1e-6`, while the public dependency normalizes by `norm + epsilon`. The patch sets `epsilon = 1e-6 / 1.02` and divides the **first** polynomial's `(a, b, c)` by `(1.02, 1.02**3, 1.02**5)`. In exact arithmetic this absorbs the missing input factor into that first polynomial. Subsequent coefficient triples are unchanged.

This is **not numerical equivalence to the existing BF16 implementation**. The input is still cast to BF16 first, but the dependency calculates its norm in FP32 and runs internal iterations in FP16, then returns BF16. The original performs normalization/iterations in BF16. Gram regrouping, restarts, the tall-matrix multiplication order, and optional kernels also alter rounding. Stability with these coefficients and `[2]` needs measurement on the target shapes. A float64 singular-value algebra check validates the scaling identity, not FP16/BF16 accuracy.

## Dependencies and execution limits

The pinned [package metadata](https://github.com/Dao-AILab/gram-newton-schulz/blob/e45d0aca7083cb275c9a303220c05c4abecd9187/pyproject.toml) requires Python >=3.10, PyTorch >=2.7.1, `quack-kernels==0.5.0`, and `nvidia-cutlass-dsl==4.5.2`. These dependency packages are required by upstream packaging even with custom kernels disabled. Use the training machine's CUDA-enabled PyTorch when installing requirements; upstream recommends `--no-build-isolation`.

The [upstream installation guidance](https://github.com/Dao-AILab/gram-newton-schulz/blob/e45d0aca7083cb275c9a303220c05c4abecd9187/README.md) targets Hopper H100 and Blackwell B200/B300, with CUDA >=12.9. Do not infer RTX 50-series custom-kernel support from the word “Blackwell.” The default PyTorch backend is the initial path for this repository's consumer GPUs; its compilation/runtime is still unverified here. For custom kernels, upstream switches to symmetric kernels only when the shorter dimension is at least 256. The patch disables the dependency's `reduce-overhead` compile mode in favor of `default`.

Optimizer work remains local. DDP owns gradient synchronization and each PP rank updates only its stage's parameters. No optimizer collective or state sharding is introduced.

## Apply and verify

```sh
git apply --check experiments/20261009-additional/05-gram-newton-schulz.patch
git apply experiments/20261009-additional/05-gram-newton-schulz.patch
```

To reverse, before changing patched lines:

```sh
git apply -R --check experiments/20261009-additional/05-gram-newton-schulz.patch
git apply -R experiments/20261009-additional/05-gram-newton-schulz.patch
```

On the unapplied baseline, `python3 experiments/20261009-additional/check_optimizer_patches.py` checks application/reversal in a temporary checkout, Python syntax, constructor wiring, and float64 singular-value recurrence algebra. Those checks passed locally. PyTorch/CUDA is unavailable here, so target-machine checks must cover actual BF16/FP32 gradients, square/tall/wide matrices, zero gradients, per-step update error, finite optimizer states, warmup restore/resume, and DDP replica agreement before a training comparison. Timing and loss/convergence are unmeasured.
