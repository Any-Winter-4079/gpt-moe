# Performance experiments — 2026-10-06

The original patches were prepared against source commit `7704a680a84be3000e407887e149914215c655c5`, which includes the grouped MoE implementation and its ReLU²/autocast correction. Batched Muon (formerly patch 05) is now incorporated into the working training code, and its experiment files have been removed. The remaining patches are stored here for testing on top of batched Muon.

The objective is to reach the same validation loss sooner by implementing the existing computation more efficiently. Layer counts, expert selection, parameter identities, learning-rate and batch schedules, data order, tokenizer, and loss objective are retained. Floating-point reduction order can change in the numerical kernels; the reports identify those cases. Existing DDP and two-stage PP interfaces are retained, but actual distributed GPU execution remains unverified.

## What to test

| Patch | Mechanism | Main measurement | Assessment |
| --- | --- | --- | --- |
| [06 — static validation masks](06-static-validation-mask.patch) | Classify exact attention tiles from their token-distance bounds during evaluation without dynamic masks | Validation time and identical validation loss | First validation-speed candidate; avoids constructing a token-pair mask for this case |
| [07 — validation autocast](07-validation-autocast.patch) | Reuse the configured BF16 autocast context during validation | Validation time and loss on the same checkpoint | One-line precision experiment; evaluation values can change |
| [01 — fused MoE routing](01-fused-moe-routing.patch) | Fuse dispatch/combine and their backwards; gather input gradients by token instead of atomic scatter | Training-step time, routing time and peak GPU memory | Promising memory-traffic hypothesis; new Triton kernels need GPU compilation and numerical checks |
| [04 — packed QKV](04-packed-qkv.patch) | Concatenate projection weights for one Q/K/V multiplication while retaining the three registered parameters | Training-step time and loss trajectory | Small implementation change; packing overhead and changed gradient summation can offset the benefit |
| [03 — diagnostic host reads](03-batch-diagnostic-host-reads.patch) | Read Q/K diagnostic scalars together, and read the three training losses together | Actual wall time; smaller possible effect on logged step time | Modest overhead reduction; Q/K reporting is outside the training timer |
| [02 — int32 shards](02-int32-shards.patch) | Keep resident shards in int32 and convert selected windows into contiguous int64 batches | CPU RAM and shard-loading time | Primarily a RAM experiment; local batch preparation was slightly slower |

Patch numbers identify the files, not their recommended order. Each matching report describes the implementation, checks, numerical limitations, and expected benefit. No GPU speedup has been measured for these patches.

## Apply one experiment

Run from the repository root after transferring this experiment folder with your usual Git workflow. For example, to test static validation masks:

```sh
git apply --check experiments/20261006-performance/06-static-validation-mask.patch
git apply experiments/20261006-performance/06-static-validation-mask.patch
```

Then use your existing training command and configuration. To return from that experiment before trying another:

```sh
git apply -R --check experiments/20261006-performance/06-static-validation-mask.patch
git apply -R experiments/20261006-performance/06-static-validation-mask.patch
```

Use the corresponding filename for another experiment. The check commands catch conflicts with later edits. Test each patch alone first; applying several at once prevents attributing a speed or loss change to one implementation. Combining successful candidates is a subsequent experiment because their performance effects need not add together.

## Comparison that answers the question

Use the same GPU environment, seed, dataset, configuration, and starting weights for the baseline and each candidate. Keep the existing 4→8 batch schedule and 8192 sequence length. Check both batch-size phases; an early improvement at four sequences does not establish the result at eight. Compare a range of steady-state steps with the same batch/window size, then the elapsed time to the same validation target, `3.28`. The normal training run is the deciding experiment; the detailed component checks in the individual reports are optional ways to investigate a failure or unexpected result.

The [uploaded earlier MoE run](https://huggingface.co/Edue3r4t5y6/nanogpt_20261006_153319_RTX-PRO-6000-95GB_BS8_SEQ8192_GA4/raw/main/log.txt) reached `3.27687502` at step **1,825**, after **400,031,744 training tokens**. Its source commit was `a8fd074eb70ef50db0f4361150e2a475a3b07259`, predating the grouped implementation used as this patch baseline. Treat that result as historical context; compare the remaining patches against the current code with batched Muon on the same machine.

The historical log reports separate costs:

| Cost | Time |
| --- | --- |
| Accumulated timed training steps | 53.44 minutes |
| Sum of 73 logged validation evaluations | 49.38 minutes |
| Kernel compilation/warmup | 28.30 minutes |

Their sum is about 131.12 minutes, before other untracked overhead. The displayed `total train time` excludes validation and compilation. Record those categories separately and record actual process wall time. Keep timer boundaries unchanged: moving work out of the reported interval is not a speedup.

## Local verification and remaining limits

The reports document CPU algebra, loader, mask, state, syntax, and graph-capture checks. Triton arithmetic was inspected or exercised through CPU stand-ins where possible; those checks do not establish CUDA lowering, kernel performance, or distributed correctness. This machine cannot run the target CUDA kernels. GPU compilation, numerical behavior, peak memory, and time to the validation target still need your tests.

The remaining patches can be applied independently on top of batched Muon. The integration check verifies that their source edits can coexist and parse together. That establishes source compatibility, not measured performance of the combined version.

## Larger research directions

The modded-nanogpt review informs the packed QKV and batched-optimizer experiments. Its record times include specific hardware and a collection of model/training changes; they are not a direct speed target for this configuration. The attention report discusses which ideas transfer and which require a separate project.

[Byte Latent Transformer](https://arxiv.org/abs/2412.09871) groups bytes into patches for a larger latent transformer and uses local byte-processing modules. It could be a useful modeling experiment, but would change the input/output representation and require a comparable metric such as bits per byte; the present token validation loss would no longer be the same benchmark.

[Block diffusion](https://arxiv.org/abs/2503.09573) generates blocks through a learned denoising process. Adding a diffusion decoder after a backbone is a possible design, but it needs a noise process, conditioning, training objective, and iterative decoding scheme. It is not an interchangeable output head for the current next-token objective. The number of diffusion iterations alone does not establish a speed benefit: the cost of each denoising pass matters. These ideas are recorded for separate modeling work rather than mixed into this implementation comparison.
