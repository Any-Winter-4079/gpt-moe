# SOAP matrix optimizer with bounded covariance dimensions

Independent patch against the working tree of 2026-10-09, including its existing whole-block activation checkpoint edits. Applying it selects `optimizer_type="soap"`, with AdamW for token/position embeddings, the output head and non-matrix parameters. Select `"muon"` or `"adamw"` to use the existing paths.

`optimizers/soap.py` vendors the actual [official SOAP implementation at commit `a1e553530fde97d0e6b307d7c82ac6d38b072340`](https://github.com/nikhilvyas/SOAP/blob/a1e553530fde97d0e6b307d7c82ac6d38b072340/soap.py), with its MIT copyright and license included in the file. It follows [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321): estimate gradient covariance matrices, project the gradient into their orthogonal bases, run bias-corrected Adam there, and project the update back. This is not a new Shampoo-like approximation or a third-party optimizer with an unverified API. No package install or runtime download is needed.

Local changes to that pinned implementation are limited to argument checks, dense 2D floating-point parameter requirements, FP32 gradients/covariance allocation, preserving FP32 optimizer state on reload, and enabling gradients for an optional closure. The reference's initialization, projections, covariance EMA, eigenbasis refresh, moment transport and weight-decay order remain intact. In particular, the first observed gradient for each matrix initializes its basis and **does not update or decay that matrix**. AdamW parameters still update on the first training step. Later SOAP weight decay happens after its adaptive update, as in the pinned source; this differs in finite arithmetic and ordering from decay-before-update AdamW.

| Configuration | Patch default | Meaning |
| --- | --- | --- |
| `optimizer_type` | `"soap"` | SOAP on matrix weights; AdamW on the remaining parameters |
| `soap_betas` | `(0.95, 0.95)` | Adam moments inside the Shampoo basis |
| `soap_lr_scale` | `0.6` | Multiplier on the existing token-based base LR schedule |
| `soap_weight_decay` | `0.01` | SOAP's decoupled weight decay |
| `soap_eps` | `1e-8` | Denominator epsilon |
| `soap_shampoo_beta` | `-1.0` | `-1` reuses the second Adam beta for covariance EMA |
| `soap_precondition_frequency` | `10` | Refresh the eigenbasis after this many SOAP updates |
| `soap_max_precond_dim` | `2048` | Omit covariance/basis allocation for larger axes |

The existing `adamw_*` values continue to configure the accompanying AdamW optimizer and base token schedule. With a base maximum of `0.005`, the SOAP maximum is `0.003`; the scale also applies to warmup, cosine and the schedule floor. These values are configurable starting points, not tuned recommendations.

The frequency controls **basis refresh**, not covariance collection: covariance EMA is updated on every present gradient. Initial bases use `eigh`; subsequent refreshes use the reference's eigenvalue reordering, second-moment permutation and one power iteration followed by QR. Axes above the dimension limit are left unprojected. For a `4096 × 2048` expert matrix with the default limit, only the `2048` axis has a covariance and basis. A limit below every matrix dimension leaves those matrices using an Adam-style update in the original basis, so lowering the limit changes the experiment. This is axis omission, not blockwise Shampoo. Dimension merging, 1D preconditioning and gradient normalization retain their reference defaults of false; bias correction remains true.

Memory is a major constraint. Each selected `m × n` matrix needs two FP32 moments (`8mn` bytes), plus an FP32 covariance and basis (`8d²` bytes) for each eligible axis `d`. For the current 18-layer, width-2048, eight-expert, hidden-width-4096 ReLU² configuration and a 2048 limit, the matrix covariances and bases alone are approximately **14.06 GiB**. The additional moment versus the current FP32 Muon state is approximately **10.13 GiB**, for about **24.19 GiB extra persistent state per full DDP replica**. These are tensor-storage estimates from parameter shapes, excluding eigensolver/QR workspaces, projected tensors, allocator overhead, gradients, weights and activations. PP owns only the matrices in its stage. BF16 weights do not shrink the FP32 SOAP states. A larger limit can increase this sharply; a smaller model should be used for the first explicitly authorized smoke check before attempting full-model timing. No fit claim is made for any GPU.

SOAP parameters are constructed in deterministic named-parameter order so state indices map consistently across restarts and ranks. Tied token/head weights remain deduplicated in AdamW. Each DDP rank operates on its synchronized local gradient tensors; each PP stage owns only its local parameters and optimizer state. There is no distributed Shampoo implementation, collective or state sharding here. Missing gradients skip the update, decay, covariance and counter. A present zero gradient follows the reference update. The generic checkpoint and warmup mechanisms save/restore all moment tensors, covariance/basis lists and per-parameter counters. The override restores FP32 values from the original saved state rather than converting already-rounded BF16 state back to FP32. The new settings are included in checkpoint configuration matching; pre-patch checkpoints do not match and should not be reused to resume this optimizer experiment.

FP32 gradient and optimizer tensors support FP32/BF16/FP16 model weights; parameter writes still round to the model dtype and there is no master-weight copy. Optimizer steps run outside the training autocast context. FP32 matrix multiplication follows the run's existing `float32_matmul_precision` setting; it does not imply an independently enforced full-mantissa matmul mode. CUDA QR/eigh behavior, runtime memory, distributed execution and convergence remain unverified. The upstream repository itself describes this implementation as preliminary.

SOAP can take a slower step than Muon or fused AdamW. Its useful outcome would be less wall time to a target held-out loss, if a convergence gain outweighs its covariance, projection and QR work. Measure both steady steps and refresh steps, peak memory, and quality at equal tokens/wall time. No throughput or convergence gain has been measured for this repository.

Local validation completed: syntax parsing; independent applicability to the live baseline; whitespace checks; isolated apply/reverse restoring the copied baseline byte-for-byte; extracted-method checks with parameter/optimizer stubs for full-model and both PP-stage ownership, embedding exclusion, disjoint parameter coverage and LR scaling. These stub checks do not execute PyTorch or a distributed backend. PyTorch was unavailable, so the optional tensor-only checker has not executed. It covers the first covariance-only step, an explicit Adam-in-eigenbasis update, covariance-size omission, FP32 state restoration for all supported model dtypes, continuation across QR refresh, missing gradients and invalid settings. It does not establish end-to-end DDP/PP or model-quality results.

Apply independently from the repository root:

```sh
git apply --check experiments/20261009-additional/09-soap.patch
git apply experiments/20261009-additional/09-soap.patch
```

Optional algebra checks, only when explicitly requested:

```sh
python experiments/20261009-additional/check_soap.py --device cpu
```

Reverse after the experiment, before applying another independent patch:

```sh
git apply --reverse --check experiments/20261009-additional/09-soap.patch
git apply --reverse experiments/20261009-additional/09-soap.patch
```
