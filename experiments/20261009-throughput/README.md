# Ten independent throughput experiments — 2026-10-09

Objective: lower **training step time / increase tokens per second** at fixed model dimensions, attention pattern, token budget, precision settings and checkpointing policy. These patches change implementations. None is a measured speedup yet.

The baseline is the current working tree at `f06e05b46c7202bb5f91859b0095ba53aab0292f`, including the pending whole-block activation-checkpointing changes in `config/training.py` and `model/gpt.py`. The production files are not modified by this experiment directory. Transfer the directory with the baseline through your normal commit/push/pull workflow.

Apply each patch independently to that baseline. These are separate from `20261009-memory` and from the four earlier performance experiments; do not stack them for the initial comparison.

| # | Patch | Cost targeted | Main qualification |
| --- | --- | --- | --- |
| 01 | `01-grouped-gemm-l2-order.patch` | Repeated weight-tile reads in grouped MoE multiplications | Reorders tile visits for cache reuse; forward and backward kernel compilation needs checking |
| 02 | `02-grouped-weight-grad-split-k.patch` | Long serial reductions in MoE weight gradients | Four parallel partial sums; extra FP32 temporary memory and changed reduction order |
| 03 | `03-expert-weight-layout.patch` | Strided expert-weight access in forward grouped GEMMs | Packs the transposed weights; packing/backward costs may outweigh the gain |
| 04 | `04-muon-foreach-momentum.patch` | Separate momentum/Nesterov launches for every matrix | Uses multi-tensor operations without changing the update equations |
| 05 | `05-polar-express-split-matmul-add.patch` | Potential defensive copies in the `aX + B @ X` operation | Activates the existing split implementation; adds an intermediate BF16 rounding |
| 06 | `06-muon-foreach-weight-update.patch` | Separate parameter-update launches after batched orthogonalization | Applies each batch of independent updates with a multi-tensor operation |
| 07 | `07-rope-strided-inputs.patch` | Explicit full copies of Q and K before rotation | Reads even/odd lanes directly; the compiler may already remove the original copies |
| 08 | `08-ddp-local-buffers.patch` | DDP forward broadcasts of deterministic model buffers | Mainly useful with multiple DDP GPUs; PP is unaffected |
| 09 | `09-single-transfer-xy.patch` | Duplicate CPU packing, pinning and H2D transfer of overlapping X/Y | Transfers one extended window; constructs contiguous X/Y on the GPU |
| 10 | `10-fused-loss-all-train-batches.patch` | Materializing logits below the current fused-loss threshold | Uses the already-integrated Liger loss for every training batch; may be slower for small batches |

I would start with **01, 04, 06 and 10**, then 03, 05, 07 and 09. Try 02 with memory headroom. Reserve 08 for a multi-GPU DDP comparison, although it also applies to the current single-GPU DDP configuration.

## Apply and reverse

From the repository root, for example:

```sh
git apply --check experiments/20261009-throughput/01-grouped-gemm-l2-order.patch
git apply experiments/20261009-throughput/01-grouped-gemm-l2-order.patch
```

After testing:

```sh
git apply -R --check experiments/20261009-throughput/01-grouped-gemm-l2-order.patch
git apply -R experiments/20261009-throughput/01-grouped-gemm-l2-order.patch
```

Substitute the filename for another experiment. A failed check means the current source no longer matches; do not force it. Revert any manual edits that overlap the patch before reversing. Test combinations only after identifying individual improvements.

## What each experiment changes

### 01 — Grouped GEMM tile order

Changes `model/grouped_linear.py`. Within each expert, visits a group of up to eight row tiles for a column tile before advancing across columns. The intended benefit is reuse of the same B/weight tiles in L2. The existing tile sizes, autotuning candidates, dot precision and per-output reduction order are retained. This applies to forward, input-gradient and weight-gradient GEMMs. Uneven final groups and empty experts retain complete tile coverage.

This follows the scheduling idea in the [Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). A different launch order can still lose on a particular GPU/cache/workload.

### 02 — Split the weight-gradient reduction

Changes `model/grouped_linear.py`. Weight-gradient GEMMs partition each expert's token-reduction dimension into four block-aligned ranges, write separate FP32 partial outputs, then sum and cast once to the original output dtype. Forward and input-gradient GEMMs keep one reduction partition. There are no atomic additions.

The hypothesis is that more independently scheduled work and shorter reductions improve utilization. It can instead become bandwidth-bound or slower from the extra reduction. The partial tensor contains **four FP32 values per weight-gradient output element**: eight times the storage of one BF16 output, or four times one FP32 output, before accounting for the final result and other live tensors. This is the highest-memory-risk patch in this set. Reduction order changes, so compare gradients and loss, not only timing.

### 03 — Expert weight layout

Changes `model/moe.py`. Builds the expert weight stack in `[expert, input, output]` order and passes a transposed view through the existing grouped-linear interface. The forward GEMM then reads a contiguous `[input, output]` matrix. Original two-dimensional expert parameters, optimizer ownership and checkpoint keys remain unchanged. Both up/down projections and all existing activation/bias paths remain available.

This still stacks weights on every forward; it is a layout experiment, not persistent packed storage. It trades the forward access pattern against packing and backward access patterns. The existing kernel autotuner keys include strides.

### 04 — Batched momentum arithmetic

Changes `optimizers/muon.py`. Groups gradient/momentum pairs by device and dtype within each optimizer parameter group. Performs the same momentum multiply, gradient addition and optional in-place Nesterov addition using `torch._foreach_*`, before the existing orthogonalization path.

Parameters with no gradient remain skipped. QKV splitting, backend coefficients, learning-rate scaling, parameter order and state keys are retained. This is separate from the already-integrated batched orthogonalization: it targets the operations **before** that computation. It adds no CPU offload.

### 05 — Split Polar Express multiply/add

Changes the default of the existing `split_baddbmm` argument in `optimizers/polar_express.py` to `True`. Each iteration uses a matrix multiplication into C followed by `C.add_(X, alpha=a)`, instead of passing X twice to `addmm`/`baddbmm`. The source already describes a possible defensive copy in the combined form.

The coefficients, iteration count and mathematical expression stay the same. The split form rounds the matrix-product result to BF16 before the addition, so bitwise equality is not expected. Whether it avoids a costly copy in the installed PyTorch version must be measured. This experiment is not a request to lower arithmetic precision elsewhere.

### 06 — Batched weight updates

Changes `optimizers/muon.py`. After each batch of Polar Express outputs is ready, uses one foreach addition for its parameter views instead of a Python loop issuing separate additions. The existing batch limit of eight and independent matrix normalization remain unchanged.

This targets the operations **after** orthogonalization and is independent of patch 04. QKV slices do not overlap. Noncontiguous views may affect whether PyTorch uses its fastest multi-tensor path. Other Muon backends keep their current per-matrix path.

### 07 — RoPE without input copies

Changes `model/position.py`. Replaces the explicit contiguous Q/K copies and pair reshapes with even/odd slices of the last dimension. The rotation arithmetic and final layout construction remain unchanged. It supports separate query/KV head counts and is shared by the FlexAttention and SDPA paths; masks and attention behavior are unchanged.

This is a small copy-elimination hypothesis. If compilation already eliminates those copies, expect little or no improvement.

### 08 — Local DDP buffers

Changes `train.py` to construct DDP with `broadcast_buffers=False`. The current model buffers are deterministic RoPE tables/frequencies and the sliding-window scalar, which every rank updates from the same schedule. Re-broadcasting them before forwards should be unnecessary for this model. Parameter/gradient synchronization remains enabled.

This depends on the current buffer inventory and identical schedule progression across ranks. It is not a general setting for models with running statistics or rank-dependent mutable buffers. PP does not use DDP and is unaffected. See [PyTorch's DDP buffer documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

### 09 — One X/Y transfer

Changes `data/loader.py`, `train.py` and `runtime/pipeline.py`. Adds a device/batch-size option to the existing loader method. GPU callers select the same sequence starts, assemble `sequence_length + 1` tokens per sequence, and transfer that window once. X and Y are constructed as contiguous GPU tensors; document IDs, when requested, are computed from the same X values on the GPU.

Token dtype remains int64. This reduces host-to-device token traffic from `2 * batch * sequence_length * 8` bytes to `batch * (sequence_length + 1) * 8` bytes. It is distinct from the earlier int32-shard experiment. CPU callers retain the existing interface, and PP still loads only on stage 0 and broadcasts targets/document IDs. Because PP evaluation shares that loader path, it also uses the combined transfer.

The tradeoff is extra GPU work to materialize the shifted tensors. No prefetch depth or data order changes, no extra CUDA stream, and no work moved outside the existing training timer. Token traffic is small compared with model computation, so the end-to-end improvement may be small.

### 10 — Fused loss at every training size

Changes `config/training.py`, setting `dense_loss_max_elements = 0`. The existing `use_liger_loss` flag remains the control; when enabled, all nonempty training batches use the integrated fused linear/cross-entropy implementation. Evaluation and sampling retain their existing paths.

The current threshold is `4 * 8192 * 50304`, so a 16,384-token microbatch currently materializes dense logits. The hypothesis is that avoiding those writes and reads helps throughput. Chunked GEMMs may instead be less efficient, and the loss/gradient accumulation order can differ. This keeps the same cross-entropy objective and padding policy, with the existing `liger-kernel==0.8.4` dependency.

Use a fresh run for this comparison: the loss threshold is part of the repository's strict resume configuration, so a checkpoint saved with the other threshold will fail that configuration check.

## Verification and later GPU checks

Local checks passed for all ten patches: independent application, candidate Python syntax, whitespace checks, and reversal to identical baseline source bytes. The production source hashes remain unchanged.

Additional local checks covered the actual tile-index calculations over 273 empty/ragged geometries, exact split-K coverage for 32 reduction geometries, RoPE arithmetic with NumPy stand-ins on three noncontiguous MHA/GQA layouts, transpose-packing algebra, and X/Y/target/document-ID equality over 12 batch/sequence combinations. These are indexing/algebra checks, not execution of PyTorch or Triton kernels. Local PyTorch/CUDA are unavailable.

Two optional component checks are provided for your later CUDA session. They execute small synthetic operations, not a pretraining run, and may compile kernels:

```sh
# after applying 01, 02 or 03; checks eager/compiled grouped forward and gradients
python experiments/20261009-throughput/check_grouped_linear.py

# after applying 04, 05 or 06; compares against Muon source at the baseline commit
python experiments/20261009-throughput/check_muon.py
```

The grouped check covers strided inputs, both weight layouts, empty/skewed expert assignments, FP32 and BF16. Its finite numerical tolerances are smoke-test thresholds, not proof of unchanged convergence. It does not replace a full MoE/model run. The Muon check includes missing gradients, QKV splitting, rectangular weights, both Nesterov settings and state reload. For patch 05 it reports weight differences without declaring them numerically equivalent, because additional BF16 rounding is intentional. Neither GPU check has been executed here.

For every candidate, compare steady steps after warmup on the same hardware and fixed batch/window size. Record tokens/s, step time, peak VRAM and losses; keep checkpointing settings fixed. Keep compilation time separate. The final decision needs the training loss/validation trajectory as well as throughput. Patch 08 additionally needs a multi-GPU DDP correctness/timing check; PP behavior and full-model compilation for these candidates remain unverified.
