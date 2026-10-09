# Nine additional independent experiments — 2026-10-09

These complete the additional precision, tokenizer, n-gram and optimizer candidates, alongside the ten patches in `../20261009-throughput` and five in `../20261009-residuals`. Each candidate is configurable and independent. None has a measured CUDA throughput or convergence improvement yet.

The baseline is the current working tree at `f06e05b46c7202bb5f91859b0095ba53aab0292f`, including the pending whole-block activation-checkpointing changes in `config/training.py` and `model/gpt.py`. These artifacts leave that baseline untouched. Commit/push/pull the baseline and experiment directories together through your usual workflow.

| Patch and notes | What it tests | How to select it after applying |
| --- | --- | --- |
| [01 — FP8 projections](01-fp8-projections.md) | Native FP8 matrix products for dense MLPs and grouped MoE experts; configurable backward precision | Set `GPTConfig.fp8_projections = "both"`; default is `"off"` |
| [02 — NVFP4 projections](02-nvfp4-projections.md) | Native SM120 E2M1 products with 16-element E4M3 block scales and an FP32 tensor scale | Set `GPTConfig.nvfp4_projections = "both"`; default is `"off"` |
| [03 — Trained byte-BPE](03-trained-byte-bpe.md) | Train/save/load a custom tokenizer and prepare separate, identity-checked shards | Prepare the artifact/shards, then set backend, artifact path and data path as documented; GPT-2 remains default |
| [04 — N-gram embeddings](04-ngram-embeddings.md) | Causal hashed bigram/trigram features added to token embeddings | Enabled by the patch; orders, table sizes, widths and gates are configurable |
| [05 — Gram Newton-Schulz](05-gram-newton-schulz.md) | A Gram-iteration backend for the existing batched Muon optimizer | Patch selects `muon_backend = "gram_newton_schulz"` |
| [06 — Fused Polar Express update](06-fused-polar-express-update.md) | Triton fusion of `B @ X + a * X` inside each existing orthogonalization iteration | Patch sets `muon_fused_polar_update = True` |
| [07 — Dion3](07-dion3.md) | Selected-row orthogonalization/error feedback with per-neuron normalization; existing AdamW for other parameters | Patch selects `optimizer_type = "dion3"` |
| [08 — Fused Lion](08-fused-lion.md) | A signed update with one fused Triton kernel per parameter and FP32 momentum | Patch selects `optimizer_type = "lion"` |
| [09 — SOAP](09-soap.md) | Adam updates in a gradient-covariance basis; existing AdamW for other parameters | Patch selects `optimizer_type = "soap"` |

Start with **06** for a focused implementation experiment, then **05**. Test **01** before **02**, initially keeping their low-precision-backward flags false. Their native kernels require the exact PyTorch 2.10.0 / Triton 3.6.0 versions documented in the notes; NVFP4 targets RTX Blackwell SM120, not a guessed Transformer Engine compatibility path. They preserve original parameter storage and saved backward tensors, so neither promises persistent VRAM savings. Quantization/reduction overhead may outweigh faster matrix instructions.

**07 and 08** are optimizer algorithm comparisons, requiring loss measurements and learning-rate tuning as well as timing. **09** is primarily a time-to-quality experiment: its covariance and basis work can slow individual steps. At the current model dimensions it adds approximately **24.19 GiB of persistent optimizer state** versus FP32 Muon, before temporary workspaces. Begin SOAP with a smaller model.

**03 and 04** can change how much learning each step achieves. N-grams add computation and parameters. A custom tokenizer changes sequence compression and the meaning of a token, so tokens/s and the existing token-normalized validation target cannot establish a fair comparison. Use identical held-out raw text, downstream quality, and byte-normalized loss where appropriate; the current validation loop does not calculate bits per byte.

## Apply one experiment at a time

From the repository root, for example:

```sh
git apply --check experiments/20261009-additional/06-fused-polar-express-update.patch
git apply experiments/20261009-additional/06-fused-polar-express-update.patch
```

Read that patch's notes for configuration and dependencies. Gram Newton-Schulz and Dion3 use pinned upstream revisions in their respective requirements changes; SOAP includes its pinned implementation and license. No dependency installation or experiment launch is performed by applying a patch.

After testing, restore any manual changes to the patch's added configuration lines, then reverse it:

```sh
git apply -R --check experiments/20261009-additional/06-fused-polar-express-update.patch
git apply -R experiments/20261009-additional/06-fused-polar-express-update.patch
```

Substitute another filename for another experiment. Do not force a failed check or stack patches for the initial comparisons. Use fresh runs: new model/optimizer settings enter strict checkpoint configuration matching, and tokenizer or architecture changes can alter parameter meaning or shape. Each note explains its resume implications.

## Verification and later measurements

All nine patches passed independent application, whitespace, Python syntax and exact-reversal checks against the baseline. Existing tracked file hashes remain unchanged. Additional dependency-free checks covered n-gram causality/boundaries, optimizer parameter ownership and constructor wiring, Gram scaling algebra, fused-kernel indexing, and tokenizer identity/configuration handling.

Optional component checkers are included and documented per experiment. CUDA/PyTorch execution, real tokenizer-library round trips, full-model compilation, DDP/PP execution, performance and convergence remain unverified locally. Static and algebra checks do not establish those results.

Keep hardware, data order, attention, token budget and checkpointing settings fixed for comparable implementation tests. Record steady step time, tokens/s, peak VRAM, compilation time separately, and validation loss. For algorithm/architecture changes also measure wall time to comparable quality. The patches retain local DDP/PP parameter ownership; the optimizer experiments do not add distributed state sharding.

Byte-latent processing is the next separate proposed experiment. It would introduce byte encoding, patch aggregation and decoding in the model, beyond the tokenizer artifact/backend integration in patch 03.
