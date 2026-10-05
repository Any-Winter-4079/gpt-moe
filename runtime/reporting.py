from __future__ import annotations

from dataclasses import fields

import torch
import torch.distributed as dist


def save_config_info(run, training_config, gpt_config) -> None:
    with open(training_config.config_filename, "w") as f:
        f.write(f"timestamp: {training_config.timestamp}\n")
        f.write(f"parallel mode: {training_config.parallel_mode}\n")
        f.write(f"world size: {run.world_size}\n\n")

        f.write("\n###################")
        f.write("\n# Training Config #")
        f.write("\n###################\n")
        for config_field in fields(training_config):
            if config_field.name in ["hf_user", "hf_token"]:
                continue
            elif config_field.name == "tokenizer":
                f.write("tokenizer: gpt2 (tiktoken)\n")
            elif config_field.name == "bf16_weights_params_and_scales":
                f.write(f"bf16_weights_params_and_scales: {list(training_config.bf16_weights_params_and_scales.keys())}\n")
            elif config_field.init is False:
                continue
            else:
                f.write(f"{config_field.name}: {getattr(training_config, config_field.name)}\n")

        f.write("\n###########################")
        f.write("\n# Derived Training Config #")
        f.write("\n###########################\n")
        for name in ("total_tokens_per_mini_step_train", "grad_accum_mini_steps", "total_tokens_per_step_val", "val_steps", "max_train_steps", "total_decay_tokens", "torch_compile_mode"):
            f.write(f"{name}: {getattr(training_config, name)}\n")

        f.write("\n################")
        f.write("\n# Model Config #")
        f.write("\n################\n")
        for name, value in gpt_config.__dict__.items():
            f.write(f"{name}: {value}\n")
    print(f"training and model configs saved to {training_config.config_filename}")


def log_parameter_counts(run, gpt_config) -> None:
    raw_gpt_model = run.raw_gpt_model
    if run.parallel_mode == "pp":
        parameter_counts = torch.tensor([
            sum(p.numel() for p in run.gpt_model.parameters() if p.requires_grad),
            raw_gpt_model.lm_head.weight.numel() if hasattr(raw_gpt_model, "lm_head") else 0,
        ], dtype=torch.long, device=run.device)
        dist.all_reduce(parameter_counts, op=dist.ReduceOp.SUM)
        total_params = parameter_counts[0].item()
        lm_head_params = parameter_counts[1].item()
    else:
        total_params = sum(p.numel() for p in run.gpt_model.parameters() if p.requires_grad)
        lm_head_params = raw_gpt_model.lm_head.weight.numel()

    if run.master_process:
        message = f"{total_params:,} parameters"
        print(message)
        run.log_buffer.append(message)

        emb_params = raw_gpt_model.transformer.wte.weight.numel()
        emb_tables = 1
        if hasattr(raw_gpt_model.transformer, "wpe"):
            emb_params += raw_gpt_model.transformer.wpe.weight.numel()
            emb_tables += 1
        if not gpt_config.use_tied_embeddings:
            emb_params += lm_head_params
            emb_tables += 1
        active_params = total_params - emb_params + emb_tables * gpt_config.d_model
        message = f"active parameters per token: {active_params:,}"
        print(message)
        run.log_buffer.append(message)
