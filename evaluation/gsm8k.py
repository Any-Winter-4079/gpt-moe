from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Optional

import torch
import torch.distributed as dist
from datasets import load_dataset


def extract_gsm8k_answer(completion: str) -> Optional[Decimal]:
    match = re.search(r"(?:^|\n)####[ \t]*(-?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)[ \t]*(?:\r?\n|$)", completion)
    return Decimal(match.group(1).replace(",", "")) if match else None


def load_gsm8k_data(run) -> None:
    config = run.training_config
    if config.gsm8k_interval != -1 and config.gsm8k_interval <= 0:
        raise ValueError("gsm8k_interval must be -1 or positive")
    if config.gsm8k_max_examples != -1 and config.gsm8k_max_examples <= 0:
        raise ValueError("gsm8k_max_examples must be -1 or positive")
    if config.gsm8k_num_shots < 0 or config.gsm8k_max_new_tokens <= 0 or config.gsm8k_examples_per_batch <= 0:
        raise ValueError("GSM8K requires nonnegative shots and positive token and batch limits")
    if not run.raw_gpt_model.is_causal:
        raise ValueError("GSM8K generation requires is_causal=True")

    prefix = "Solve each problem. End your answer with a line containing #### followed by the final number.\n\n"
    if config.gsm8k_num_shots:
        train_dataset = load_dataset("openai/gsm8k", "main", split="train")
        for example in train_dataset.select(range(config.gsm8k_num_shots)):
            # retain the worked solution while removing calculator annotations
            answer = re.sub(r"<<.*?>>", "", example["answer"])
            prefix += f"Question: {example['question']}\nAnswer: {answer}\n\n"

    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    total_examples = len(test_dataset)
    if config.gsm8k_max_examples > 0:
        total_examples = min(total_examples, config.gsm8k_max_examples)
    indices = range(total_examples) if run.parallel_mode == "pp" else range(run.rank, total_examples, run.world_size)
    run.local_gsm8k_examples = []
    for index in indices:
        example = test_dataset[index]
        prompt = f"{prefix}Question: {example['question']}\nAnswer:"
        answer = extract_gsm8k_answer(example["answer"])
        if answer is None:
            raise ValueError(f"GSM8K test example {index} has no numeric reference answer")
        run.local_gsm8k_examples.append((run.tokenizer.encode(prompt), answer))

    max_prompt_length = torch.tensor(
        max((len(ids) for ids, _ in run.local_gsm8k_examples), default=0),
        dtype=torch.long, device=run.device,
    )
    dist.all_reduce(max_prompt_length, op=dist.ReduceOp.MAX)
    block_size = run.raw_gpt_model.flex_block_size if run.raw_gpt_model.use_flex_attention else 1
    required_length = math.ceil((max_prompt_length.item() + config.gsm8k_max_new_tokens) / block_size) * block_size
    if required_length > run.raw_gpt_model.max_seq_len:
        raise ValueError("GSM8K prompts and generation exceed max_seq_len; reduce gsm8k_num_shots or gsm8k_max_new_tokens")

    if run.master_process:
        message = (
            f"GSM8K: {total_examples:,} test examples | first {config.gsm8k_num_shots} train examples as shots | "
            f"greedy decoding | max new tokens: {config.gsm8k_max_new_tokens} | final answer: #### number"
        )
        print(message)
        run.log_buffer.append(message)


def evaluate_gsm8k(run) -> float:
    config = run.training_config
    raw_gpt_model = run.raw_gpt_model
    eager_gpt_model = raw_gpt_model._orig_mod if hasattr(raw_gpt_model, "_orig_mod") else raw_gpt_model
    tokenizer = run.tokenizer
    examples = run.local_gsm8k_examples
    score_on_this_rank = run.parallel_mode == "ddp" or run.rank == run.world_size - 1
    total_correct = 0
    total_missing = 0
    total_examples = 0
    was_training = run.gpt_model.training
    run.gpt_model.eval()

    try:
        with torch.inference_mode(), run.ctx:
            for start in range(0, len(examples), config.gsm8k_examples_per_batch):
                batch = examples[start:start + config.gsm8k_examples_per_batch]
                prompt_lengths = [len(ids) for ids, _ in batch]
                block_size = raw_gpt_model.flex_block_size if raw_gpt_model.use_flex_attention else 1
                alloc_len = math.ceil((max(prompt_lengths) + config.gsm8k_max_new_tokens) / block_size) * block_size
                input_ids = torch.full(
                    (len(batch), alloc_len), raw_gpt_model.pad_token_id, dtype=torch.long, device=run.device,
                )
                for i, (ids, _) in enumerate(batch):
                    input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=run.device)
                lengths = torch.tensor(prompt_lengths, dtype=torch.long, device=run.device)
                rows = torch.arange(len(batch), device=run.device)
                generated_ids = [[] for _ in batch]
                completions = ["" for _ in batch]
                finished = [False for _ in batch]

                for _ in range(config.gsm8k_max_new_tokens):
                    # fixed shapes avoid recompilation as tokens arrive; causal attention excludes right padding
                    logits = run.forward(eager_gpt_model, input_ids, attn_mask=None, ignore_doc_mask=True, document_ids=None)
                    if score_on_this_rank:
                        next_ids = logits[rows, lengths - 1, :tokenizer.n_vocab].argmax(dim=-1)
                    else:
                        next_ids = torch.empty(len(batch), dtype=torch.long, device=run.device)
                    del logits
                    if run.parallel_mode == "pp":
                        dist.broadcast(next_ids, src=run.world_size - 1)
                    input_ids[rows, lengths] = next_ids
                    lengths += 1

                    # both pipeline stages see the same tokens and make the same stopping decision
                    for i, token in enumerate(next_ids.tolist()):
                        if finished[i]:
                            continue
                        if token == raw_gpt_model.eos_token_id:
                            finished[i] = True
                            continue
                        generated_ids[i].append(token)
                        completions[i] = tokenizer.decode(generated_ids[i])
                        # wait for the end of the answer line so a multi-token number is not cut short
                        finished[i] = bool(re.search(r"(?:^|\n)####[^\n]*\n|\nQuestion:", completions[i]))
                    if all(finished):
                        break

                if score_on_this_rank:
                    for completion, (_, reference) in zip(completions, batch):
                        prediction = extract_gsm8k_answer(completion)
                        total_correct += prediction is not None and prediction == reference
                        total_missing += prediction is None
                        total_examples += 1
                    print(f"[rank {run.rank}] GSM8K processed {total_examples:,} / {len(examples):,}: acc: {total_correct / total_examples:.4f}")

            totals = torch.tensor([total_correct, total_missing, total_examples], dtype=torch.long, device=run.device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            correct, missing, count = totals.tolist()
            accuracy = correct / count
            if run.master_process:
                message = f"GSM8K correct: {correct:,} / {count:,} | missing or malformed answers: {missing:,} | acc: {accuracy:.4f}"
                print(message)
                run.log_buffer.append(message)
        return accuracy
    finally:
        run.gpt_model.train(was_training)
