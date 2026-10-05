from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.nn import functional as F


def get_sample_token_count(step: int, base: int = 5, step_interval: int = 1000, max_tokens: int = 50) -> int:
    return min(base + (step // step_interval) * base, max_tokens)


def sample(
        run,
        sample_sequences: Sequence[str],
        max_new_tokens: int = 5,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        ) -> None:
    gpt_model = run.gpt_model
    raw_gpt_model = run.raw_gpt_model
    tokenizer = run.tokenizer
    device = run.device
    ctx = run.ctx
    rank = run.rank
    world_size = run.world_size
    master_process = run.master_process
    log_buffer = run.log_buffer
    gpt_model.eval()
    with torch.inference_mode():
        # convert all prompts to ids,
        # e.g., 'AGI is' to [4760, 40, 318]
        # 'AGI is not' to [4760, 40, 318, 407]
        # and truncate them leaving max_new_tokens for generation
        max_allowed_input_len = raw_gpt_model.max_seq_len - max_new_tokens
        initial_input_ids_list = [
            tokenizer.encode(sequence)[:max_allowed_input_len] 
            for sequence in sample_sequences
        ]

        # find the max input length (in ids) in this batch
        max_input_len = max(len(ids) for ids in initial_input_ids_list)

        # get a length cap to pre-allocate the tensor while not needing to go to max_seq_len
        alloc_len = max_input_len + max_new_tokens

        # rounding up to a nice multiple for better tensor cores / GPU efficiency
        round_multiple = (raw_gpt_model.flex_block_size if raw_gpt_model.use_flex_attention else 8)
        alloc_len = math.ceil(alloc_len / round_multiple) * round_multiple
        alloc_len = min(alloc_len, raw_gpt_model.max_seq_len)

        # pre-allocate tensor of size len(sample_sequences), alloc_len and fill with padding 
        # to significanly boost performance (versus a new tensor size every generation)!
        generated_sequences = torch.full(
            (len(sample_sequences), alloc_len),
            raw_gpt_model.pad_token_id,
            dtype=torch.long,
            device=device
        )

        # then replace the first padding tokens of each pre-allocated sequence with their original tokens
        for i, seq_ids in enumerate(initial_input_ids_list):
            generated_sequences[i, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)

        # track actual (non-padding) sequence lengths
        actual_sequence_lengths = torch.tensor([len(seq) for seq in initial_input_ids_list], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # then, for each new token to generate, update the mask by effectively comparing if 0, ..., alloc_len < the non-padded length for each sequence
            # using unsqueeze(0) to add a new dimension of size 1 at the beginning to give size (1, alloc_len)
            # and unsqueeze(1) to add a new dimension of size 1 at the end to give size (len(sample_sequences), 1)
            # attn_mask = torch.arange(alloc_len, device=device).unsqueeze(0) < actual_sequence_lengths.unsqueeze(1)
            
            with ctx:
                # and predict, resulting in (len(sample_sequences), seq_len, vocab_size)
                # NOTE: for decoding and right padding, tokens cannot attend to padding anyway
                # so we can skip passing an attn_mask
                eager_gpt_model = raw_gpt_model._orig_mod if hasattr(raw_gpt_model, "_orig_mod") else raw_gpt_model
                logits = run.forward(eager_gpt_model, generated_sequences, attn_mask=None, ignore_doc_mask=True, document_ids=None)
            
            if run.parallel_mode == "ddp" or rank == world_size - 1:
                # of which we take the vocab_size values for each sequence's continuation to the last non-pad token,
                # resulting in len(sample_sequences), vocab_size
                last_logits = logits[torch.arange(len(sample_sequences), device=device), actual_sequence_lengths - 1, :]

                if temperature == 0.0:
                    # then if temperature is 0, we cannot divide by 0, so take the max logit from vocab_size
                    next_token_ids = torch.argmax(last_logits, dim=-1, keepdim=True)
                else:
                    # else, divide by the temperature
                    temp_adjusted_logits = last_logits / temperature

                    # apply top_k filtering
                    if top_k is not None and top_k > 0:
                        # topk returns the k largest elements of the given input tensor along a given dimension
                        # resulting in len(sample_sequences), vocab_size
                        sorted_values, _ = torch.topk(temp_adjusted_logits, min(top_k, temp_adjusted_logits.size(-1)), dim=-1, largest=True, sorted=True)
                        # get the k-th largest logit of each sequence (i.e., last in sorted_values)
                        # and add a new dimension of size 1 at the end to give size (len(sample_sequences), 1)
                        sequences_k_th_logit = sorted_values[:, -1].unsqueeze(1)
                        # mask out everything less than the top_k logit
                        temp_adjusted_logits[temp_adjusted_logits < sequences_k_th_logit] = -float('Inf')

                    # apply top_p (nucleus) filtering
                    if top_p is not None and top_p < 1.0:
                        # apply softmax to get probabilities for the continuation to the last non-pad token
                        # resulting in (still) len(sample_sequences), vocab_size
                        probs = F.softmax(temp_adjusted_logits, dim=-1)
                        # sort probabilities in descending order
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                        # obtain the cumulative sums of the probabilities sorted in descending order
                        # e.g., cumulative probs of [0.3, 0.6, 0.8, 1.0] for [0.3, 0.3, 0.2, 0.2]
                        # with size len(sample_sequences), vocab_size
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        if (run.parallel_mode == "ddp" and master_process) or (run.parallel_mode == "pp" and rank == world_size - 1):
                            message = "first 2 sequences cumulative probs:"
                            print(message)
                            log_buffer.append(message)
                            for i in range(min(2, cumulative_probs.size(0))):
                                values = cumulative_probs[i, :10]
                                message = f"\tseq {i}: {values.tolist()}"
                                print(message)
                                log_buffer.append(message)
                        # obtain the indices of logits to remove,
                        # initially excluding the logit that makes the cumulative sum match top_p,
                        # as we then shift to the right one position
                        # cumulative_probs [0.3, 0.6, 0.8, 1.0], top_p 0.75 would thus give (for a single sequence):
                        # [False, False, True, True] which shifted to the right is [False, False, False, True]
                        # while cumulative_probs [0.3, 0.6, 0.8, 1.0], top_p 0.8:
                        # [False, False, True, True] which shifted to the right is [False, False, False, True],
                        # being in both instances the smallest possible set that has at least top_p cumulative probability
                        sorted_indices_to_remove = cumulative_probs >= top_p
                        # shift the indices to the right one position
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        # replacing the last index value, now in the first position, to False
                        sorted_indices_to_remove[..., 0] = False
                        mask = sorted_indices_to_remove.to(torch.bool)

                        scatter_mask = torch.zeros_like(temp_adjusted_logits, dtype=torch.bool)
                        scatter_mask = scatter_mask.scatter(dim=-1, index=sorted_indices, src=mask)
                        if (run.parallel_mode == "ddp" and master_process) or (run.parallel_mode == "pp" and rank == world_size - 1):
                            for i in range(min(2, scatter_mask.size(0))):
                                masked_count = scatter_mask[i].sum().item()
                                total_count = scatter_mask.size(1)
                                #print(f"sequence {i} mask: {masked_count}/{total_count} logits masked")
                                if masked_count == total_count:
                                    message = f"all logits masked for sequence {i}!"
                                    print(message)
                                    log_buffer.append(message)
                        temp_adjusted_logits = temp_adjusted_logits.masked_fill(scatter_mask, -float('inf'))
                    
                    probs = F.softmax(temp_adjusted_logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1)

            else:
                next_token_ids = torch.empty((len(sample_sequences), 1), dtype=torch.long, device=device)
            # both stages must consume the same next token on the following forward pass
            if run.parallel_mode == "pp":
                dist.broadcast(next_token_ids, src=world_size - 1)

            # replace the 'actual_sequence_length' position of each sequence with the selected token
            generated_sequences[torch.arange(len(sample_sequences), device=device), actual_sequence_lengths] = next_token_ids[:, 0]
            # increase the actual, non-padded sequence endings
            actual_sequence_lengths += 1

        # and decode, ignoring above 50256
        local_messages = []
        if run.parallel_mode == "ddp" or rank == world_size - 1:
            if run.parallel_mode == "pp":
                local_messages.extend(log_buffer)
                log_buffer.clear()
            for i in range(len(sample_sequences)):
                decoded = tokenizer.decode([
                    token for token in generated_sequences[i, :actual_sequence_lengths[i]].tolist()
                    if token < raw_gpt_model.pad_token_id])
                message = f"[rank {rank}] seq {i} >>> {decoded}"
                print(message)
                local_messages.append(message)

        all_messages = [None for _ in range(world_size)]
        dist.all_gather_object(all_messages, local_messages)

        if master_process:
            for rank_messages in all_messages:
                log_buffer.extend(rank_messages)
        dist.barrier()
    gpt_model.train()
