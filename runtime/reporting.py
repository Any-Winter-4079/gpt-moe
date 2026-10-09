from __future__ import annotations

import subprocess
from dataclasses import fields
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist


def flush_log(run) -> None:
    if run.master_process and run.log_buffer:
        with open(run.training_config.log_filename, "a") as file:
            for line in run.log_buffer:
                file.write(line + "\n")
        run.log_buffer.clear()


def log_source_code(log_buffer: List[str]) -> None:
    source_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        changes = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_root, text=True, stderr=subprocess.DEVNULL,
        )
        log_buffer.extend([f"git commit: {commit}", f"git working tree modified: {bool(changes.strip())}"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        log_buffer.append("git commit: unavailable")

    log_buffer.append("hardware information: startup snapshot")
    for command in (["nvidia-smi"], ["lscpu"], ["free", "-b"]):
        log_buffer.append(f"$ {' '.join(command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            log_buffer.append(result.stdout.rstrip("\n"))
            if result.stderr:
                log_buffer.append(result.stderr.rstrip("\n"))
            if result.returncode:
                log_buffer.append(f"exit code: {result.returncode}")
        except (OSError, subprocess.TimeoutExpired) as error:
            log_buffer.append(f"unavailable: {error}")

    # free can report host RAM, so also record the exposed cgroup membership and limits
    for filename in ("/proc/self/cgroup", "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max"):
        log_buffer.append(f"{filename}:")
        try:
            log_buffer.append(Path(filename).read_text(encoding="utf-8").rstrip("\n"))
        except OSError as error:
            log_buffer.append(f"unavailable: {error}")

    # read only Python files in the source directories, without traversing datasets or environments
    source_dirs = ("", "checkpointing", "config", "data", "evaluation", "model", "optimizers", "runtime", "sampling", "schedules")
    for directory in source_dirs:
        for path in sorted((source_root / directory).glob("*.py")):
            log_buffer.extend([
                "=" * 100,
                f"source: {path.relative_to(source_root).as_posix()}",
                "=" * 100,
                path.read_text(encoding="utf-8"),
            ])


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
    inactive_expert_params = 0
    if gpt_config.use_moe:
        inactive_expert_params = sum(
            p.numel()
            for block in raw_gpt_model.transformer.h
            for expert in block.mlp.experts[block.mlp.top_k:]
            for p in expert.parameters() if p.requires_grad
        )
    if run.parallel_mode == "pp":
        parameter_counts = torch.tensor([
            sum(p.numel() for p in run.gpt_model.parameters() if p.requires_grad),
            inactive_expert_params,
        ], dtype=torch.long, device=run.device)
        dist.all_reduce(parameter_counts, op=dist.ReduceOp.SUM)
        total_params = parameter_counts[0].item()
        inactive_expert_params = parameter_counts[1].item()
    else:
        total_params = sum(p.numel() for p in run.gpt_model.parameters() if p.requires_grad)

    if run.master_process:
        message = f"{total_params:,} parameters"
        print(message)
        run.log_buffer.append(message)

        # the full output projection stays active, including when tied to the input embedding
        active_params = total_params - inactive_expert_params
        if not gpt_config.use_tied_embeddings:
            active_params -= raw_gpt_model.transformer.wte.weight.numel() - gpt_config.d_model
        if hasattr(raw_gpt_model.transformer, "wpe"):
            active_params -= raw_gpt_model.transformer.wpe.weight.numel() - gpt_config.d_model
        message = f"active parameters per token: {active_params:,}"
        print(message)
        run.log_buffer.append(message)
