# Experiment 04: configurable causal n-gram embeddings

Apply `04-ngram-embeddings.patch` independently to the current working tree at `f06e05b46c7202bb5f91859b0095ba53aab0292f`, including the pending whole-block activation-checkpointing changes. This is an architecture experiment aimed at time to a target validation loss. Additional lookups, projections, optimizer state and gradient communication can reduce tokens/s; no convergence or speed benefit is established.

## Source and adaptation

The official [modded-nanogpt n-gram table](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/track_1_short/ngram_table.py) uses hashed bigram and trigram rows in an 84,602,880-row, 768-dimensional table, an independent sign-pool lookup, sharding and a custom sparse row optimizer. Its [model](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/track_1_short/model/gpt.py) injects their signed sum through learned gates at several layers. Source revision checked on 2026-10-09: `4ea6b937337a4889b8cfe3f38a93d120048d8f71`.

This patch adapts the causal hashed-lookup idea. Each requested order has a separate ordinary PyTorch embedding table, an optional dimension projection, and a learned scalar gate; their outputs are added once to the initial token representation. It uses a bounded int64 polynomial hash and masks incomplete or crossing-document windows. It does not reproduce the upstream wrapped-int32 hashes, sign pool, repeated layer injections, giant sharded table, sparse optimizer or tuned learning-rate multipliers.

## Configuration

All settings are in `GPTConfig`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `use_ngram_embeddings` | `True` | Enable this experiment; `False` restores baseline construction and forward behavior |
| `ngram_orders` | `[2, 3]` | N-gram lengths, each at least two |
| `ngram_dims` | `[128, 128]` | Embedding width for each order |
| `ngram_table_sizes` | `[65521, 65521]` | Hash-table rows for each order |
| `ngram_gate_init` | `0.1` | Initial value of each independent learned scalar gate |

The three lists must have matching nonempty lengths. For example, `[2, 3, 4]`, `[128, 64, 64]`, and `[65521, 32749, 32749]` add three channels. Dimensions and row counts can differ between channels. Each dimension unequal to `d_model` gets a bias-free `Linear(dim, d_model)`; equality uses `Identity`, so setting a dimension to `d_model` removes that projection. Tables and projections receive the existing normal initialization with standard deviation 0.02. Gates use the separate scalar initialization above. A zero gate preserves the initial input representation, but table/projection gradients then start at zero until the gate moves.

For channel `j`, the contribution at token position `t` is:

```text
x[t] += valid_j[t] * gate_j * projection_j(table_j[hash_j[t]])
```

Absolute positional embeddings are added after this sum; RoPE/NoPE continue through their existing paths. Channels are independent and their contributions are summed without an extra normalization.

## Hash and boundary semantics

Tokens first become int64. Starting at `h = 0`, visit `x[t], x[t-1], ..., x[t-n+1]` and apply:

```text
h = (65599 * h + token_id + 1) % table_rows
```

Table row counts and vocabulary size must be positive and smaller than `2**31`. With valid token IDs, every intermediate is at most `140,874,927,177,601`, below `2**48`, independent of the order. There is no intentional signed overflow and no conversion to int32. Hash collisions share a learned row and are part of the experiment.

Each sequence row starts fresh. A channel contributes zero until the complete n-token history is available, including when the requested order exceeds the sequence length. Following the loader's convention, EOS belongs to the document it terminates: an n-gram may end on EOS, but no later n-gram spans across it. Every supplied `document_ids` transition also restarts history. Padding IDs contribute zero and restart the following token's history. Repeated document labels cannot bypass an intervening boundary because comparisons use cumulative boundary counts.

These boundaries always apply to n-gram features, including validation and sampling calls with `ignore_doc_mask=True`; that flag continues to control attention only. Right-padded sampling is causal because future padding never enters a valid earlier lookup. Computation uses only the current input window, without cached history across batches or across truncated generation windows. The module uses token IDs for padding boundaries, matching this repository's padding convention.

## Parameters, optimizers and parallelism

For table sizes `V_j`, widths `D_j`, and hidden width `D`, the additional parameter count is:

```text
sum(V_j * D_j) + sum(D_j * D for channels with D_j != D) + number_of_channels
```

At the current `d_model=2048`, the defaults add **17,297,666 parameters**: 16,773,376 table entries, 524,288 projection weights, and two gates. The active-parameter logger counts one row per channel plus the complete projections and gates: **524,546 additional active parameters per interior token**. This is the full-history convention; masked short prefixes have no n-gram contribution.

Tables are explicitly excluded from Muon and enter decayed AdamW. Projections follow the existing 2D Muon/AdamW selection, and gates use non-decayed AdamW. Embedding gradients are dense: these small tables are replicated in DDP, and all-reduced like other parameters. Full FP32 weights, gradients and two optimizer-sized moment tensors would occupy approximately 264 MiB at the defaults, excluding activations and workspace. Actual optimizer choices and precision settings change that estimate. Table sizes from the upstream sharded implementation are unsuitable here: this patch does not shard or offload them.

The registered module lives under `transformer.ngram`. Pipeline stage 0 owns it; stage 1 deletes it when retaining its model half. The existing pipeline boundary carries the resulting hidden state, without extra channels or metadata. The existing PP model constructor still briefly builds the complete model on CPU before discarding the other stage. The parameter logger subtracts unused lookup rows on the master after the existing cross-stage total reduction.

Ordinary registered parameters are included in DDP/PP checkpoints and full-model exports, and the existing export merge retains stage-0 n-gram keys. `GPTConfig` already enters the saved resume/export configuration. Resume requires the same n-gram settings and an experiment checkpoint; existing strict checks intentionally reject old baseline checkpoints rather than migrate them.

## Application and local verification

From the repository root:

```sh
git apply --check experiments/20261009-additional/04-ngram-embeddings.patch
git apply experiments/20261009-additional/04-ngram-embeddings.patch
```

Reverse with:

```sh
git apply -R --check experiments/20261009-additional/04-ngram-embeddings.patch
git apply -R experiments/20261009-additional/04-ngram-embeddings.patch
```

Restore any later manual edits to the patch's configuration lines before reversing. Apply one experiment at a time and start a fresh comparison run.

Completed checks:

- Independent patch application against the live working tree and exact byte-for-byte reversal in a scratch copy.
- Python AST parsing of all four affected source files.
- The actual hash/mask statements extracted from the new module ran in a small stdlib integer-matrix harness against an independent window oracle: 6,720 cases covering sequence lengths 1–40, orders 2/3/4/16, multiple batch rows, EOS, padding, explicit document changes, table size one, large IDs and table sizes near `2**31`, and order greater than sequence length.
- Prefix results were invariant under truncating future tokens and future document labels. Explicit EOS-ending and padding examples matched the documented valid positions.
- The maximum hash-intermediate bound and default parameter totals were checked using Python integer arithmetic.

The integer harness checks indexing and hash algebra, not PyTorch operators, autograd or GPU execution. Local PyTorch/CUDA are unavailable. Runtime gradients, eager/compiled equivalence, mixed precision, actual DDP/1F1B execution, checkpoint round trips, VRAM and throughput remain unverified. A later authorized GPU comparison should check those paths with nonzero gates, short and padded prompts, and multiple n-gram orders, then compare validation loss at equal tokens and wall-clock time to the same loss.
