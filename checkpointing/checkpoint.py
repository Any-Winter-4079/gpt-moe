from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file, save_model, load_file

from model.gpt import GPT


def keep_latest_checkpoints(checkpoint_dir: str, max_checkpoints_to_keep: int, log_buffer: List[str]) -> None:
    files_by_step = {}
    for filename in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"(?:model|training_state)_step_(\d+).*\.(?:safetensors|pt)", filename)
        if match is not None:
            files_by_step.setdefault(int(match.group(1)), []).append(filename)

    for old_step in sorted(files_by_step)[:-max_checkpoints_to_keep]:
        for filename in files_by_step[old_step]:
            path = os.path.join(checkpoint_dir, filename)
            os.remove(path)
            message = f"removed checkpoint file: {path}"
            print(message)
            log_buffer.append(message)


def save_checkpoint(
        checkpoint_dir: str,
        gpt_model: nn.Module,
        log_buffer: List[str],
        step: int,
        rng_states_by_rank: List[Dict[str, Tensor]],
        val_loss: Optional[float],
        train_loss: float,
        train_tokens_processed: int,
        total_train_t: float,
        total_val_t: float,
        total_sample_t: float,
        total_hellaswag_t: float,
        total_t: float,
        best_val_loss: float,
        epoch: int,
        current_shard_idx: int,
        grad_accum_mini_steps_per_shard_counter: int,
        optimizers: Dict[str, torch.optim.Optimizer],
        parallel_mode: str,
        world_size: int,
        rank: int,
        resume_config: Dict[str, Any],
        schedule_state: Dict[str, Any],
        best_improvement_count: int,
        allow_val: bool,
        ) -> str:

    # create the model path by adding its timestamp, and train and val losses, if available
    parts = ["model", f"step_{step:07d}"]
    parts.append(f"val_{val_loss:.4f}" if val_loss is not None else "val_na")
    parts.append(f"train_{train_loss:.4f}")
    stage_suffix = f"_stage_{rank}" if parallel_mode == "pp" else ""
    checkpoint_name = "_".join(parts) + stage_suffix + ".safetensors"
    safetensors_path = os.path.join(checkpoint_dir, checkpoint_name)

    # save the model locally
    model_to_save = gpt_model.module._orig_mod if parallel_mode == "ddp" else gpt_model._orig_mod
    save_model(model_to_save, safetensors_path)
    message = f"saved model weights locally in: {safetensors_path}"
    print(message)
    log_buffer.append(message)

    # gather the training state
    training_state = {
        'format_version': 2,
        'parallel_mode': parallel_mode,
        'world_size': world_size,
        'rank': rank,
        'resume_config': resume_config,
        'schedule_state': schedule_state,
        'best_improvement_count': best_improvement_count,
        'allow_val': allow_val,
        'step': step,
        'rng_states_by_rank': rng_states_by_rank,
        'train_tokens_processed': train_tokens_processed,
        'total_train_t': total_train_t,
        'total_val_t': total_val_t,
        'total_sample_t': total_sample_t,
        'total_hellaswag_t': total_hellaswag_t,
        'total_t': total_t,
        'best_val_loss': best_val_loss,
        'epoch': epoch,
        'current_shard_idx': current_shard_idx,
        'grad_accum_mini_steps_per_shard_counter': grad_accum_mini_steps_per_shard_counter,
        'optimizer_state_dicts': {k: optimizer.state_dict() for k, optimizer in optimizers.items()}
    }

    # create the training state path
    state_path = os.path.join(checkpoint_dir, f"training_state_step_{step:07d}{stage_suffix}.pt")

    # save the training state locally
    torch.save(training_state, state_path)
    message = f"saved training state locally in: {state_path}"
    print(message)
    log_buffer.append(message)
    return safetensors_path


def load_checkpoint(training_config, gpt_config, rank: int, world_size: int, resume_config: Dict[str, Any]) -> Tuple[
    int,                                # start_step
    Tensor,                             # torch_rng_state_cpu
    Tensor,                             # torch_rng_state_cuda
    int,                                # train_tokens_processed
    float,                              # total_train_t
    float,                              # total_val_t
    float,                              # total_sample_t
    float,                              # total_hellaswag_t
    float,                              # total_t
    float,                              # best_val_loss
    int,                                # epoch
    int,                                # current_shard_idx
    int,                                # grad_accum_mini_steps_per_shard_counter
    Dict[str, Any],                     # schedule_state
    int,                                # best_improvement_count
    bool,                               # allow_val
    Dict[str, Any],                     # optimizer_state_dicts
    "GPT",                              # gpt_model
]:
    hub_repo_id = training_config.hub_repo_id
    resume_checkpoint_path = training_config.resume_checkpoint_path
    resume_state_dict_path = training_config.resume_state_dict_path
    hf_token = training_config.hf_token
    if not resume_checkpoint_path or not resume_state_dict_path:
        raise ValueError("both resume checkpoint paths must be configured")
    if training_config.parallel_mode.lower() == "pp":
        if "{rank}" not in resume_checkpoint_path or "{rank}" not in resume_state_dict_path:
            raise ValueError("pipeline resume paths must contain {rank} for the stage number")
        resume_checkpoint_path = resume_checkpoint_path.format(rank=rank)
        resume_state_dict_path = resume_state_dict_path.format(rank=rank)

    model_path = resume_checkpoint_path if os.path.isfile(resume_checkpoint_path) else hf_hub_download(
        repo_id=hub_repo_id, filename=resume_checkpoint_path, token=hf_token, repo_type="model"
    )
    training_state_path = resume_state_dict_path if os.path.isfile(resume_state_dict_path) else hf_hub_download(
        repo_id=hub_repo_id, filename=resume_state_dict_path, token=hf_token, repo_type="model"
    )

    training_state = torch.load(training_state_path, map_location="cpu")
    if training_state.get('format_version') != 2:
        raise ValueError("checkpoint format version 2 is required")
    if training_state['parallel_mode'] != training_config.parallel_mode.lower():
        raise ValueError("checkpoint parallel mode differs from the current configuration")
    if training_state['world_size'] != world_size:
        raise ValueError("checkpoint GPU count differs from the current run")
    if training_config.parallel_mode.lower() == "pp" and training_state['rank'] != rank:
        raise ValueError("checkpoint stage number differs from this rank")
    # allow the requested AdamW moment precision to override the checkpoint setting
    checkpoint_resume_config = {
        **training_state['resume_config'],
        'training': {
            **training_state['resume_config']['training'],
            'use_bf16_adamw_moments': training_config.use_bf16_adamw_moments,
        },
    }
    if checkpoint_resume_config != resume_config:
        raise ValueError("checkpoint model or training configuration differs from the current run")

    # load weights
    model_state_dict = load_file(model_path, device='cpu')

    stage_index = rank if training_config.parallel_mode.lower() == "pp" else None
    gpt_model = GPT(gpt_config, training_config, stage_index=stage_index)
    if gpt_model.use_tied_embeddings:
        embedding_key = "transformer.wte.weight"
        head_key = "lm_head.weight"
        if embedding_key not in model_state_dict and head_key in model_state_dict:
            model_state_dict[embedding_key] = model_state_dict[head_key]
        elif head_key not in model_state_dict and embedding_key in model_state_dict:
            model_state_dict[head_key] = model_state_dict[embedding_key]
    gpt_model.load_state_dict(model_state_dict, strict=True)

    rng_states_by_rank = training_state['rng_states_by_rank']
    expected_count = world_size if training_config.parallel_mode.lower() == "ddp" else 1
    if len(rng_states_by_rank) != expected_count:
        raise ValueError("checkpoint RNG state count differs from the current run")
    rng_state = rng_states_by_rank[rank if training_config.parallel_mode.lower() == "ddp" else 0]

    return (
        training_state['step'] + 1,
        rng_state['cpu'],
        rng_state['cuda'],
        training_state['train_tokens_processed'],
        training_state['total_train_t'],
        training_state['total_val_t'],
        training_state['total_sample_t'],
        training_state['total_hellaswag_t'],
        training_state['total_t'],
        training_state['best_val_loss'],
        training_state['epoch'],
        training_state['current_shard_idx'],
        training_state['grad_accum_mini_steps_per_shard_counter'],
        training_state['schedule_state'],
        training_state['best_improvement_count'],
        training_state['allow_val'],
        training_state['optimizer_state_dicts'],
        gpt_model,
    )


def save_stage_weights_for_export(gpt_model: nn.Module, export_dir: str, step: int, rank: int) -> str:
    path = os.path.join(export_dir, f"stage_weights_step_{step:07d}_stage_{rank}.safetensors")
    if os.path.exists(path):
        raise FileExistsError(path)
    save_model(gpt_model._orig_mod, path)
    return path


def export_full_model_ddp(gpt_model: nn.Module, resume_config: Dict[str, Any], export_dir: str, step: int, current_window_size: int, checkpoint_model_path: Optional[str]) -> str:
    path = os.path.join(export_dir, f"full_model_step_{step:07d}.safetensors")
    if os.path.exists(path):
        raise FileExistsError(path)
    if checkpoint_model_path is not None:
        try:
            os.link(checkpoint_model_path, path)
        except OSError:
            shutil.copyfile(checkpoint_model_path, path)
    else:
        save_model(gpt_model.module._orig_mod, path)
    with open(os.path.join(export_dir, f"full_model_step_{step:07d}_config.json"), "w") as file:
        json.dump({**resume_config, "sliding_window_size": current_window_size}, file, indent=2)
    return path


def export_full_model_pp(stage_paths: List[str], resume_config: Dict[str, Any], export_dir: str, step: int, current_window_size: int) -> str:
    if len(stage_paths) != 2:
        raise ValueError("full-model export requires both pipeline stage files")
    path = os.path.join(export_dir, f"full_model_step_{step:07d}.safetensors")
    if os.path.exists(path):
        raise FileExistsError(path)
    split = resume_config["model"]["n_layers"] // 2
    merged_weights = {}
    for stage_index, stage_path in enumerate(stage_paths):
        for key, value in load_file(stage_path, device="cpu").items():
            if stage_index == 1 and key.startswith("transformer.h."):
                parts = key.split(".", 3)
                parts[2] = str(int(parts[2]) + split)
                key = ".".join(parts)
            if key in merged_weights:
                raise ValueError(f"duplicate model weight during pipeline export: {key}")
            merged_weights[key] = value
    save_file(merged_weights, path)
    with open(os.path.join(export_dir, f"full_model_step_{step:07d}_config.json"), "w") as file:
        json.dump({**resume_config, "sliding_window_size": current_window_size}, file, indent=2)
    return path
