# Experiment 03: train and reuse a custom byte-BPE tokenizer

This independent patch adds a configurable Hugging Face `tokenizers` byte-BPE backend while retaining GPT-2/tiktoken as the default. It leaves the readable `tokenizer/bpe.py` teaching implementation and the existing FineWeb download script intact. Apply against the current working tree, including its existing activation-checkpointing changes:

```sh
git apply --check experiments/20261009-additional/03-trained-byte-bpe.patch
git apply experiments/20261009-additional/03-trained-byte-bpe.patch
```

The new preparation commands use local UTF-8 JSONL: one document per line, for example `{"text": "A document\nwith a second line."}`. They do not download data. The compiled Rust BPE trainer learns merges from a configurable prefix of training documents; it does not use the slow teaching implementation to train a large vocabulary.

## Prepare the artifact and fresh shards

These are commands for a later, explicitly authorized preparation run. They were not executed while creating this patch. The environment needs `tokenizers==0.21.4` (pinned in the patch's requirements) and NumPy for shard preparation, alongside the repository's existing dependencies.

First establish disjoint raw training and validation document lists. Keep benchmark/test text out of tokenizer fitting. If reusing FineWeb, recover the intended raw document split before fitting; the old script's first token-count shard does not define the same raw split after changing tokenizers. Do not decode the old GPT-2 shards and assume that their token boundary is an independent document split.

```sh
python -m tokenizer.prepare train \
  --train-jsonl /local/corpus/train-000.jsonl /local/corpus/train-001.jsonl \
  --output ./tokenizer_artifacts/domain-32k.json \
  --vocab-size 32768 --min-frequency 2 --max-documents 100000

python -m tokenizer.prepare shards \
  --tokenizer ./tokenizer_artifacts/domain-32k.json \
  --train-jsonl /local/corpus/train-000.jsonl /local/corpus/train-001.jsonl \
  --val-jsonl /local/corpus/validation.jsonl \
  --output-dir ./data/domain_byte_bpe_32k \
  --shard-size 100000000
```

Supply the intended complete training list to `shards`; `--max-documents` limits tokenizer fitting only. The target vocabulary includes 256 byte symbols and EOS, excludes the model padding row, and must be 257–65535. The actual fitted vocabulary can be smaller than the target. The saved JSON includes the learned vocabulary, merges, byte pre-tokenizer and decoder. ByteLevel uses its regex with `add_prefix_space=False`, preserves whitespace and Unicode without normalization, and prevents merges across its pre-tokenized boundaries. This is a new tokenizer, not GPT-2 vocabulary compatibility.

Shard preparation inserts the artifact's EOS once before each document, writes the existing flat `uint16` `.npy` format, and handles documents spanning multiple shards. Train and validation stay separate. It refuses existing output directories and identical input paths shared between splits. It cannot detect duplicate document content across different files; curate the source splits accordingly. Choose production shard sizes large enough for the configured loader batches; tiny tail shards still inherit the existing loader's minimum-size constraints.

A completed directory contains `tokenizer_metadata.json` with artifact identity and the exact shard filename list. Custom training rejects legacy directories without that manifest, different artifacts or library versions, and extra/missing `.npy` files. It also rejects a custom manifest when GPT-2 is selected. The manifest is published only after both splits finish; an interrupted preparation needs a new output directory. This is an identity check, not a per-shard content checksum: do not replace individual files under existing names or manually copy a manifest to GPT-2 shards.

For a GPT-2 control using the same raw documents, run the `shards` command with `--backend gpt2`, omit `--tokenizer`, and use a different fresh output directory. Existing GPT-2 shards remain usable with the default backend.

## Select the experiment

Set these existing/new fields in `TrainingConfig` after applying the patch:

```python
tokenizer_backend: str = "byte_bpe"
tokenizer_path: str = "./tokenizer_artifacts/domain-32k.json"
tokenizer_vocab_multiple: int = 128
data_path: str = "./data/domain_byte_bpe_32k"

# per-token NLL depends on the tokenizer; disable the GPT-2 stopping target
val_target: float = -1.0
train_val_margin: float = -float("inf")
```

The last two settings disable the old loss target while enabling regular validation under the existing gating logic. Choose appropriate sequence length, token budget and validation coverage for the experiment. No training launch is part of this patch preparation.

For this backend, `train.py` sets `GPTConfig.eos_token_id` from the JSON, reserves `pad_token_id = tokenizer.n_vocab`, and rounds the embedding/output vocabulary up to `tokenizer_vocab_multiple`, including that padding row. For an actual vocabulary of 32768 and a multiple of 128, the model vocabulary is 32896. Corpus IDs and pad fit `uint16`; extra padded output rows only belong to the model. The training loss retains the repository's existing full padded output vocabulary.

Training, HellaSwag, GSM8K and sampling use the same resolved tokenizer object. Sampling excludes padding/alignment rows from generated IDs, matching GSM8K's existing vocabulary slicing. GPT stores its configured EOS so GSM8K can stop on the correct ID. Literal `<|endoftext|>` in ordinary text is encoded as text; shard creation alone explicitly adds document EOS. Inference prompts receive no implicit EOS or padding from the tokenizer.

## Checkpoint and export identity

Custom resume/export configuration records SHA-256 of the exact tokenizer JSON bytes, backend/library version, vocabulary size, EOS and pad IDs. Existing strict resume comparison therefore rejects a different artifact even if its filename and vocabulary size match. Moving an identical JSON to another local path is allowed; resume still obeys the repository's other configuration checks, including its existing `data_path` check.

The exact artifact bytes are saved as `tokenizer.json` beside run logs, enabled checkpoints and enabled full-model exports. They also travel through the existing checkpoint/log folder upload workflow if the user later runs it. The patch does not launch uploads or alter the upload script. Preserve the JSON with weights and export configuration, make it available on every worker, and select it for resume/inference; weights alone do not identify the token-to-ID mapping. Changing the tokenizer requires newly tokenized shards and a fresh model, not resuming or reusing old embedding/output weights. GPT-2 defaults retain their previous model IDs and resume dictionary, so pre-patch GPT-2 checkpoints do not gain a new mandatory identity field.

## Compare and verify

The old validation target `3.28` and per-token perplexity are not comparable across tokenizers. Compare the same held-out raw documents, downstream benchmark settings, and raw byte coverage. A normalized language-model metric should sum held-out token NLL and divide by the corresponding UTF-8 byte count (and by `ln(2)` for bits per byte), with identical EOS/context conventions. The current validation loop does not compute that normalization; a fixed token prefix can cover different raw text. A fixed training token budget also changes the amount of raw text seen. Measure these effects separately before claiming quality or throughput improvements.

Checks executed during patch preparation:

- Patch apply against the current working tree, apply/reverse in a scratch copy, and exact baseline restoration.
- Python syntax for every patched Python source.
- Dependency-free checker: invalid backend/path rejection, GPT-2 legacy acceptance, required custom manifest, artifact mismatch and mixed-file rejection, and derived EOS/pad/vocabulary alignment over boundary values.

```sh
python experiments/20261009-additional/check_trained_byte_bpe.py
```

The additional command below is supplied but **not run**. It requires the pinned tokenizer package and NumPy; it trains only on three synthetic strings and checks save/load identity, unseen Unicode/whitespace/literal EOS round trips, and tiny multishard packing. It makes no model calls or network requests.

```sh
python experiments/20261009-additional/check_trained_byte_bpe.py --runtime
```

Local Python lacks `tokenizers`, `tiktoken`, NumPy and PyTorch, so the real library round trip, shard encoding, CUDA training, benchmark generation and distributed resume/export paths remain untested. No dependencies were installed, datasets downloaded, corpus fitting/retokenization performed, or performance results fabricated.

The pinned API was checked against Hugging Face's official [v0.21.4 Python bindings](https://github.com/huggingface/tokenizers/blob/v0.21.4/bindings/python/py_src/tokenizers/__init__.pyi), [BPE trainer contract](https://github.com/huggingface/tokenizers/blob/v0.21.4/bindings/python/py_src/tokenizers/trainers/__init__.pyi), and [literal special-token encoding behavior](https://github.com/huggingface/tokenizers/blob/v0.21.4/tokenizers/src/tokenizer/added_vocabulary.rs).
