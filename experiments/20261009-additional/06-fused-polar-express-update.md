# 06 — Fused Polar Express update

This independent patch replaces the last matrix update in each Polar Express iteration with a Triton matmul and epilogue: `C = a * X + B @ X`. It is built against the current working tree, including the existing uncommitted changes to `config/training.py` and `model/gpt.py`. Production Python files were not edited to create it.

`TrainingConfig.muon_fused_polar_update` defaults to **True in this patch**, so applying it enables the experiment for the configured Muon/Polar Express optimizer. Set it to **False** for the original runtime computation. The flag flows through GPT optimizer construction into each Muon parameter group and the existing PE bucket call. Standalone Muon and PE function arguments still default to False. SVD and Newton–Schulz calls retain their existing arguments and computation.

From the repository root:

```sh
git apply --check experiments/20261009-additional/06-fused-polar-express-update.patch
git apply experiments/20261009-additional/06-fused-polar-express-update.patch
```

To remove this patch:

```sh
git apply --reverse experiments/20261009-additional/06-fused-polar-express-update.patch
```

The patch is independent of the earlier throughput patch `05-polar-express-split-matmul-add.patch`. The new `fused_update=True` argument takes precedence over `split_baddbmm`; when fused_update is False, the existing split flag behaves exactly as before. Apply experiments individually. Adjacent config/signature edits in other patches may conflict when combined.

The kernel uses BF16 inputs, an FP32 `tl.dot` accumulator, and an FP32 `tl.fma` epilogue before one BF16 output store. It handles `[M, N]` and `[batch, M, N]` inputs with square `B`, explicit batch/row/column strides, and masked tails. Positive matrix dimensions and a separately allocated, nonoverlapping output are its contract. The existing PE path already allocates separate X/C buffers and swaps them only after a full update; it also retains its contiguous normalized X. Transposes and sliced noncontiguous inputs/outputs are supported by the kernel wrapper. This follows the [Triton matmul pattern](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), [BF16 dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html), and [fused multiply-add API](https://triton-lang.org/main/python-api/generated/triton.language.fma.html).

The existing five coefficient triples, five iterations, norm/safety factor/epsilon, transpose rule, BF16 PE output, bucket size eight, QKV handling, momentum, Nesterov and `sqrt(max(rows, cols))` update scaling are preserved. Three tile configurations are autotuned; cold-start compilation/tuning is additional cost. The wrapper follows the project's existing direct Triton launch pattern inside compiled PE. PyTorch documents support for [user-defined Triton kernels and configs/key autotuning under torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html).

The performance hypothesis is that a custom output stage can avoid the existing addmm/baddbmm alias-copy path noted in the source, while eliminating the additional launch and intermediate BF16 C traffic of the split matmul/add alternative. That copy and any speedup must be established by profiling the actual CUDA/PyTorch build. A custom matmul can also be slower than cuBLAS. No throughput or memory improvement is claimed from the local checks.

Numerically, the new kernel rounds the FP32 matmul-plus-epilogue result to BF16 once. The split alternative first stores `B @ X` in BF16 and rounds again after the addition, so its finite-precision result can differ. The default addmm/baddbmm path already expresses the same fused mathematical operation, but different reduction order, tiling, FMA details, and library reduction precision can still prevent bitwise agreement. PyTorch documents both [non-associativity and BF16 reduced-precision GEMM reductions](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms). The patch changes no global precision settings. Differences can propagate across all five PE iterations and subsequent optimizer steps; training equivalence remains unverified.

The new flag is included in the existing strict `resume_config` comparison. Checkpoints must have the same flag value. Checkpoints made before this field existed fail that exact comparison, even with the experiment disabled; no checkpoint migration is included. This avoids silently resuming across a changed optimizer computation.

Local verification passed: Python parsing/bytecode compilation for all five patched files and the checker; live-tree `git apply --check`; scratch apply matching the candidate byte for byte; reverse application restoring the captured base byte for byte; AST comparison showing disabled PE is the original function apart from its new argument/branch; unchanged coefficients; actual wrapper execution with lightweight tensor metadata checking 2D/3D grid and stride forwarding; and host-only tile/mask coverage emulation for 18 shape/configuration combinations. Torch and Triton are absent locally. **CUDA compilation, numerical checks, profiling, and training have not been run.**

The optional CUDA checker is ready for a separately approved run after applying this patch:

```sh
python experiments/20261009-additional/check_fused_polar_update.py
```

It compares isolated updates against FP64 arithmetic on the same BF16 inputs with `rtol=0.016, atol=0.001`, including small/tail dimensions, batches and noncontiguous storage. It checks unchanged inputs, fullgraph compiled PE, zero updates, the real Muon bucket path, QKV, the eight-item chunk boundary, absent gradients, both Nesterov settings and state reload. Full PE and weight differences are printed with finite-value checks rather than treated as proof of equivalent training. It does not measure speed; warmed optimizer timings and a CUDA trace on the configured 2048/4096 matrix sizes are still needed.
