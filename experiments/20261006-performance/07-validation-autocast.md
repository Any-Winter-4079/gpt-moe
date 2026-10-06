# 07 — Validation autocast

This patch changes one line in `evaluation/validation.py` to reuse `run.ctx` during validation. With the current configuration, eligible operations use BF16 autocast. It follows the existing precision settings and covers both DDP and pipeline parallelism. The configured FP32 loss and FP32 validation accumulation remain in place.

Apply on top of the incorporated batched Muon implementation:

```sh
git apply --check experiments/20261006-performance/07-validation-autocast.patch
git apply experiments/20261006-performance/07-validation-autocast.patch
```

Reverse with:

```sh
git apply -R experiments/20261006-performance/07-validation-autocast.patch
```

Measure validation time after its first compiled pass. Autocast changes evaluation arithmetic, so compare the validation loss on the same checkpoint and validation tokens to distinguish precision differences from training changes. Compilation may run again for the new evaluation dtype. Syntax, applicability, and compatibility with the remaining patches were checked locally; CUDA performance and the loss difference require the rented GPU.
