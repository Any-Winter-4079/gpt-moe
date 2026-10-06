# 03: batch diagnostic host reads

Baseline: `7704a680a84be3000e407887e149914215c655c5`.
Patch: `03-batch-diagnostic-host-reads.patch`, independently applicable with `git apply`.
Files: `model/attention.py`, `train.py`.

The Q/K logger currently converts four CUDA scalars to Python floats per layer. With the default eight layers, that is 32 individual scalar reads per logged step. This patch retains the existing per-layer reductions and clamp operations, stacks their scalar results, and reads them with one `tolist()`. It also reads the existing three-element training-loss tensor once instead of converting its components separately.

Expected benefit is modest: fewer blocking host reads and more opportunity to enqueue the small Q/K reduction kernels before waiting. The Q/K diagnostics run **after `end_train_t`**, so their improvement would affect actual wall time, not the reported training-step duration or its tokens/second. The loss read remains inside the existing training timer; its contribution should be much smaller. Timer boundaries and timing aggregation are unchanged. No speedup has been measured.

Diagnostic text, per-step frequency, per-layer clamp rounding, and the original Python `max(0.0, ...)` behavior for negative and NaN results are preserved. The stack promotes mixed floating dtypes without changing their already-reduced values for the tested fp16/bf16/fp32/fp64 types. Optimizer arithmetic, loss accumulation and scaling, validation and stopping comparisons, and distributed collectives are unchanged. DDP still logs Q/K values on rank 0; existing PP still gathers each stage's diagnostic string. All parameters of each local stage reside on one device, as required by the current runtime.

The extra stack allocates a small device tensor and may add a packing kernel. On an unconstrained host, that overhead could offset some or all of the reduced-copy benefit. CPU evidence cannot establish CUDA transfer cost or multi-GPU performance.

Local verification used CPU PyTorch 2.9.0 via:

```sh
/var/folders/5j/rxlcfqfs01d82ch5y38n7rt80000gn/T/gpt-moe-opcheck-ntzu20_x/bin/python /tmp/gpt-moe-runtime-performance-cuc8m1d0/check_diagnostics.py
```

- 167 exact old/new diagnostic-string comparisons passed: empty/disabled layers, varied head counts and caps, all-negative values, signed zero, NaN/Inf, and mixed fp16/bf16/fp32/fp64 layers. Parameters remained unchanged and no gradients were created.
- 103 training-loss comparisons passed with identical Python float values, including signed zero, subnormals, NaN/Inf, and large finite values.
- CPU profiler counted 32 baseline `aten::_local_scalar_dense` calls versus zero for the new eight-layer diagnostic function; this verifies removal of individual scalar extraction calls, not CUDA speed.
- Both complete changed Python files compiled. `git apply --check --whitespace=error-all` passed, and applying to a separate baseline copy reproduced the tested candidate byte for byte. Production files remain identical to the baseline.

GPU verification remains required: compare steady-state wall time as well as the unchanged logged training timer, check diagnostic text against baseline, and exercise DDP with 1/2/many GPUs plus existing two-stage PP. Keep logging enabled during comparison. CUDA transfer behavior, GPU numerical execution, distributed execution, and speedup are unverified.
