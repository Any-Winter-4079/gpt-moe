"""Check experiment 03 after applying it; --runtime additionally needs tokenizers and numpy."""

import argparse
import ast
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def check_static(root):
    from tokenizer.loader import load_tokenizer, validate_dataset_tokenizer

    for name in ("config/training.py", "model/gpt.py", "runtime/reporting.py", "sampling/sample.py",
                 "train.py", "tokenizer/loader.py", "tokenizer/prepare.py"):
        ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
    expect_value_error(lambda: load_tokenizer("wrong", ""))
    expect_value_error(lambda: load_tokenizer("byte_bpe", ""))
    expect_value_error(lambda: load_tokenizer("gpt2", "custom.json"))
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        custom = {"backend": "byte_bpe", "sha256": "artifact-a", "n_vocab": 300}
        validate_dataset_tokenizer(str(folder), {"backend": "gpt2"})
        expect_value_error(lambda: validate_dataset_tokenizer(str(folder), custom))
        shard = folder / "corpus_train_000000.npy"
        shard.touch()
        (folder / "tokenizer_metadata.json").write_text(
            json.dumps({"tokenizer": custom, "shards": [shard.name]}), encoding="utf-8",
        )
        validate_dataset_tokenizer(str(folder), custom)
        expect_value_error(lambda: validate_dataset_tokenizer(str(folder), {**custom, "sha256": "artifact-b"}))
        expect_value_error(lambda: validate_dataset_tokenizer(str(folder), {"backend": "gpt2"}))
        (folder / "old_gpt2_train.npy").touch()
        expect_value_error(lambda: validate_dataset_tokenizer(str(folder), custom))
    tree = ast.parse((root / "train.py").read_text(encoding="utf-8"))
    configure = next(node for node in tree.body if isinstance(node, ast.If)
                     and any(isinstance(item, ast.Attribute) and item.attr == "vocab_size" for item in ast.walk(node)))
    code = compile(ast.Module(body=[configure], type_ignores=[]), "train.py", "exec")
    for n_vocab in (257, 32768, 50257, 65535):
        for multiple in (1, 64, 128, 256):
            model = SimpleNamespace()
            exec(code, {"gpt_config": model, "tokenizer": SimpleNamespace(n_vocab=n_vocab, eot_token=0),
                        "training_config": SimpleNamespace(tokenizer_backend="byte_bpe", tokenizer_vocab_multiple=multiple)})
            assert model.eos_token_id == 0 and model.pad_token_id == n_vocab
            assert model.vocab_size > model.pad_token_id
            assert model.vocab_size % multiple == 0
            assert model.vocab_size - multiple <= model.pad_token_id
    print("PASS: syntax, backend errors, manifest/artifact rejection, derived EOS/pad and vocabulary alignment")


def check_runtime():
    import numpy as np
    from tokenizer.loader import load_tokenizer, validate_dataset_tokenizer
    from tokenizer.prepare import train_tokenizer, tokenize_corpus

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        train_path = folder / "train.jsonl"
        val_path = folder / "val.jsonl"
        train_texts = ["banana banana bandana\n" * 8, "hello world", "literal <|endoftext|>"]
        val_texts = ["unseen: español, 中文, 🧪\n\t", ""]
        for path, texts in ((train_path, train_texts), (val_path, val_texts)):
            path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
        artifact = folder / "tokenizer.json"
        train_tokenizer(SimpleNamespace(
            train_jsonl=[str(train_path)], output=str(artifact), vocab_size=300,
            min_frequency=2, max_documents=100,
        ))
        tokenizer, identity = load_tokenizer("byte_bpe", str(artifact))
        assert identity["eos_token_id"] == 0
        assert identity["pad_token_id"] == tokenizer.n_vocab
        for text in train_texts + val_texts + [" a  b\r\n", "<|endoftext|>"]:
            ids = tokenizer.encode(text)
            assert tokenizer.decode(ids) == text
            assert tokenizer.eot_token not in ids
        copy = folder / "saved.json"
        tokenizer.save(str(copy))
        loaded, loaded_identity = load_tokenizer("byte_bpe", str(copy))
        assert loaded_identity == identity
        assert loaded.encode(val_texts[0]) == tokenizer.encode(val_texts[0])
        copy.write_text("changed", encoding="utf-8")
        expect_value_error(lambda: tokenizer.save(str(copy)))
        output = folder / "shards"
        tokenize_corpus(SimpleNamespace(
            backend="byte_bpe", tokenizer=str(artifact), train_jsonl=[str(train_path)],
            val_jsonl=[str(val_path)], output_dir=str(output), shard_size=7,
        ))
        validate_dataset_tokenizer(str(output), identity)
        for split, texts in (("train", train_texts), ("val", val_texts)):
            arrays = [np.load(path) for path in sorted(output.glob(f"corpus_{split}_*.npy"))]
            assert all(array.dtype == np.uint16 and 0 < len(array) <= 7 for array in arrays)
            expected = [token for text in texts for token in [tokenizer.eot_token] + tokenizer.encode(text)]
            assert np.concatenate(arrays).tolist() == expected
    print("PASS: tiny BPE save/load, Unicode/literal EOS round trips, artifact preservation, multishard document packing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime", action="store_true", help="train BPE on three synthetic strings and check tiny local shards")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    check_static(args.source_root)
    if args.runtime:
        check_runtime()
