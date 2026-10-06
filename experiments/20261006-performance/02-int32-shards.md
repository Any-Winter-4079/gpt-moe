# Int32 resident shards with packed window gathering

Baseline: `7704a680a84be3000e407887e149914215c655c5`. Patch changes only `data/loader.py`.

Keep materialized CPU shards in int32 and share their NumPy allocation through `torch.from_numpy`. Prefetch uses the same load method. Gather each selected `seq_len + 1` window once, then cast x and y separately to contiguous int64. This keeps the existing NumPy int32 narrowing semantics, including uint32 overflow, and the existing public batch interface.

**Primary benefit: 50% less resident shard storage.** A current and prefetched 100M-token shard take 800 MB together instead of 1.6 GB, per loader/rank. With equally sized train and validation shards, that is 1.6 GB instead of 3.2 GB per DDP rank. PP retains its single master loader. These are tensor-storage bytes; process overhead and transient buffers are additional. GPU batch memory and transfer sizes stay the same.

Local synthetic CPU results (PyTorch 2.1.2, NumPy 1.24.4, one CPU thread):

| Measurement | Baseline | Patch |
| --- | ---: | ---: |
| Resident storage, 16M-token shard | 128 MB | 64 MB |
| Warm-cache 16M uint16 shard load, median of 6 | 9.687 ms | 6.535 ms |
| 8 × 8192 batch, no document IDs | 33.548 µs | 35.175 µs |
| 8 × 8192 batch, with document IDs | 160.596 µs | 161.263 µs |

The batch microbenchmark repeatedly selects the same batch and measures CPU preparation only. Its small regression is explicit; this is primarily a RAM/startup/shard-transition experiment, with low expected step-time impact unless host memory pressure is significant. No CUDA, pinned-memory transfer, DDP process group, PP scheduler, training, or rented-GPU timing was executed.

CPU equivalence passed **674 configurations / 16,816 batch pairs**: worlds 1/2/3/8 and every rank; train plus shuffled/ordered validation; documents on/off; uint16/int32/uint32/int64 source arrays; shard and epoch transitions; exact-multiple shard lengths and their existing short final batches; reset; five checkpoint/resume positions; unchanged global torch RNG; 8192-token sequences; and unchanged max-batch-8 consumption followed by the 4→8 caller slice. Ordered validation prefixes were checked across all rank counts. Both completed-prefetch and forced blocking-load paths were exercised. Outputs are equal, separate, contiguous int64 tensors.

The implementation uses existing NumPy/PyTorch CPU operations and adds no dependencies, flags, checkpoint fields, or distributed communication. Local harnesses: `/private/tmp/gpt-moe-loader-sym8d6y7/check_loader.py` and `bench_loader.py`.

**Existing limitation:** a slow prefetch can publish stale tokens after a blocking fallback, causing a later shard index to consume the previous shard's data. This was deterministically reproduced in both baseline and patch with `check_prefetch_race.py`; reset also retains the existing untagged-worker risk. The patch leaves that separate concurrency bug unchanged. Equivalence checks do not establish correctness under arbitrary worker timing.

From the repository root at the baseline, apply independently:

```sh
git apply --check experiments/20261006-performance/02-int32-shards.patch
git apply experiments/20261006-performance/02-int32-shards.patch
```

Rollback after the experiment:

```sh
git apply --reverse --check experiments/20261006-performance/02-int32-shards.patch
git apply --reverse experiments/20261006-performance/02-int32-shards.patch
```

Forward application, reverse application, and Python syntax were checked in isolated copies. Production source files were not modified.
