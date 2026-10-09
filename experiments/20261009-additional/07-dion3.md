# 07 — Dion3 optimizer experiment

Independent patch: `07-dion3.patch`. Apply to the current baseline, including the pending whole-block checkpointing edits. This is an **optimizer algorithm change**, so compare convergence at equal token budgets as well as optimizer time and end-to-end step time.

The patch imports the actual public `Dion3` from [Microsoft/Dion revision 7692479288f928a3d61d1cc58137b4dbe4f71000](https://github.com/microsoft/dion/blob/7692479288f928a3d61d1cc58137b4dbe4f71000/dion/dion3.py), pinned in `requirements.txt`. In that revision `Dion3` is an alias of `NorDion2`. It uses Dion2's accumulated-gradient row selection/error feedback followed by NorMuon's per-neuron normalization; the library's internal algorithm identifier remains `"nordion2"`. The run's separate optimizer/checkpoint key is correctly named `"dion3"`.

## Controls

| TrainingConfig field | Patch default | Meaning |
| --- | --- | --- |
| `optimizer_type` | `"dion3"` | Select Dion3 plus the existing AdamW; `"muon"` selects the original optimizer |
| `dion3_lr_scale` | `0.75` | Multiplies the base scheduled AdamW LR before the library's dimension adjustment |
| `dion3_fraction` | `0.25` | Fraction of rows selected in each matrix or Q/K/V block; `1.0` selects all rows |
| `dion3_mu` | `0.95` | Decay/error-feedback factor applied to the selected accumulated-gradient rows |
| `dion3_beta2` | `0.95` | Per-neuron second-moment smoothing |
| `dion3_weight_decay` | `0.0` | Matrix weight decay; matches the baseline Muon's zero matrix decay |
| `dion3_eps` | `1e-6` | Orthogonalization normalization epsilon; matches the baseline Polar Express epsilon |
| `dion3_adjust_lr` | `"rms_norm"` | `"rms_norm"`, `"spectral_norm"`, or `None` |
| `dion3_use_triton` | `False` | Switch from compiled PyTorch Polar Express to upstream Triton Polar Express |
| `dion3_triton_post_ortho` | `False` | Enable the dependency's Triton selected-row weight update |

The native row selection dimension is fixed by Dion3: rows preserve the per-neuron normalization geometry. It selects `ceil(fraction * rows)`, at least one row, using accumulated-gradient L1 norms and `topk`; it does not select a random subset. A fraction of 1 still retains Dion3's normalization/update rule and does not recover the original Muon.

The [dimension scaling](https://github.com/microsoft/dion/blob/7692479288f928a3d61d1cc58137b4dbe4f71000/dion/megabatch_base.py) is `0.2 * sqrt(max(rows, columns))` for `rms_norm`, `sqrt(rows / columns)` for `spectral_norm`, and 1 for `None`. Thus the default `0.75 * 0.2 = 0.15` preserves the baseline's scalar LR factor at the full matrix/block dimensions. This is an initial comparison setting, not a tuned Dion3 learning rate, and does not make the updates equivalent.

## Parameter ownership and resume

The current parameter partition is retained: 2D matrices except token embeddings and `lm_head` move to Dion3; those excluded weights and all 0D/1D parameters retain the existing AdamW decay groups. Expert matrices remain separate parameters. The baseline's shape-based `(3d, d)` split is represented with the library's public `num_heads=3` parameter group. This preserves that existing heuristic, including its application to any matrix with that shape; other matrices stay whole. Current attention projections are separate Q/K/V parameters.

Dion3 groups retain deterministic `named_parameters()` order and keep original parameter objects, enabling stable state IDs on fresh process resume. The training loop schedules both AdamW and Dion3 under their proper keys. All new controls enter the strict resume configuration, and the library's optimizer state is saved/restored by the existing checkpoint and warmup paths. Start a fresh run when changing optimizer type or adding this configuration schema.

`distributed_mesh=None` is deliberate for both supported modes: DDP already reduces gradients and every replica executes the same local update; PP owns different local stage parameters, so those ranks must not be treated as replicated matrices. The patch adds no optimizer collectives, compression hook, or state sharding. Therefore it does not provide Dion's distributed optimizer communication savings.

Inspection of the [selection implementation](https://github.com/microsoft/dion/blob/7692479288f928a3d61d1cc58137b4dbe4f71000/dion/dion2.py) found no RNG calls on this path. Different DDP rank seeds do not affect selection. `topk` ties and floating-point kernels still require a replica-equality check on the target GPU/software combination; cross-device bitwise determinism is not established here.

The [native Dion3 state implementation](https://github.com/microsoft/dion/blob/7692479288f928a3d61d1cc58137b4dbe4f71000/dion/nordion2.py) stores full momentum and a rows-by-one variance buffer in the parameter dtype, including BF16 when parameters are BF16. Normalization computes with FP32 intermediates and writes variance back to that stored dtype. This is not an FP32-master-state optimizer. The native state loader's cast to parameter dtype agrees with that design; its override also restores persistent device LR tensors. Memory includes the extra per-neuron variance and upstream same-shape megabatch temporaries, which are not capped to the baseline Muon's eight-matrix batches.

## Dependencies and verification

Upstream [base requirements](https://github.com/microsoft/dion/blob/7692479288f928a3d61d1cc58137b4dbe4f71000/requirements_dion.txt) are NumPy and PyTorch >=2.7.1. The patch uses compiled PyTorch Polar Express by default. Enabling either Triton control requires the target training stack's compatible Triton. It does not require Gram Newton-Schulz or CuTeDSL. Upstream's BF16 Polar Express uses the same five coefficient triples as this repository, but its arithmetic order and Dion3's selected matrix inputs differ.

```sh
git apply --check experiments/20261009-additional/07-dion3.patch
git apply experiments/20261009-additional/07-dion3.patch
```

To reverse, before changing patched lines:

```sh
git apply -R --check experiments/20261009-additional/07-dion3.patch
git apply -R experiments/20261009-additional/07-dion3.patch
```

On the unapplied baseline, `python3 experiments/20261009-additional/check_optimizer_patches.py` passed patch apply/reverse, syntax, and stubbed execution of the actual optimizer-configuration method. It checks disjoint/exhaustive parameter ownership, QKV groups, stable order, separate optimizer names, local process ownership, and controls for full/DDP and stage-local parameter lists, including empty AdamW groups. Constructor keywords were also checked against the pinned source.

PyTorch/CUDA is unavailable locally. Actual native optimizer steps, state round trips, target-shape compilation, DDP replica agreement, PP execution, memory, timing, and convergence remain unverified. Source inspection of state dtypes is not a substitute for a target-machine save/load comparison.
