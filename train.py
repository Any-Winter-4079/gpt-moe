import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import math
import time
from dataclasses import asdict
from types import SimpleNamespace

import torch
import triton
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from config.model import GPTConfig
from config.training import TrainingConfig
from model.gpt import GPT
from model.attention import qk_scale_debug_string
from optimizers.muon import Muon
from data.loader import DataLoader
from schedules.token import get_next_update_tokens
from sampling.sample import get_sample_token_count, sample
from evaluation.hellaswag import load_hellaswag_data, evaluate_hellaswag_standard
from evaluation.validation import evaluate_validation
from checkpointing.checkpoint import (
    export_full_model_ddp, export_full_model_pp, keep_latest_checkpoints,
    load_checkpoint, save_checkpoint, save_stage_weights_for_export,
)
from runtime.precision import convert_to_bf16, precision_context
from runtime.reporting import check_finite_tensors, flush_log, save_config_info, log_parameter_counts
from runtime.warmup import kernel_warmup
from runtime.pipeline import PipelineRuntime

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device_type = "cuda"
device = f"cuda:{local_rank}"
torch.cuda.set_device(device)
init_process_group(backend="nccl", device_id=local_rank)
master_process = rank == 0

training_config = TrainingConfig()
parallel_mode = training_config.parallel_mode.lower()
assert parallel_mode in ("ddp", "pp"), "parallel_mode must be 'ddp' or 'pp'"
if parallel_mode == "pp":
    assert world_size == 2, "Pipeline parallelism requires torchrun --nproc_per_node=2"
assert training_config.checkpoint_interval == -1 or training_config.checkpoint_interval > 0
assert training_config.checkpoint_best_improvement_interval == -1 or training_config.checkpoint_best_improvement_interval > 0
assert training_config.full_model_export_interval == -1 or training_config.full_model_export_interval > 0
assert training_config.max_checkpoints_to_keep > 0
effective_world_size = world_size if parallel_mode == "ddp" else 1
training_config.resolve(effective_world_size)
if parallel_mode == "pp":
    assert training_config.grad_accum_mini_steps >= world_size, "The pipeline schedule requires at least two microbatches"
    assert all(0 < size <= training_config.gpu_batch_size_train for size in training_config.batch_size_schedule_values), \
        "Scheduled microbatch sizes must fit gpu_batch_size_train"

gpu_batch_size_train = training_config.gpu_batch_size_train
gpu_batch_size_val = training_config.gpu_batch_size_val
seq_len_train = training_config.seq_len_train
seq_len_val = training_config.seq_len_val
max_train_steps = training_config.max_train_steps
grad_accum_mini_steps = training_config.grad_accum_mini_steps
val_steps = training_config.val_steps
batch_size_schedule_keys = training_config.batch_size_schedule_keys
batch_size_schedule_values = training_config.batch_size_schedule_values
lr_schedule = training_config.lr_schedule
tokenizer = training_config.tokenizer
sample_interval = training_config.sample_interval
sample_sequences = training_config.sample_sequences
hellaswag_interval = training_config.hellaswag_interval
val_interval = training_config.val_interval
val_target = training_config.val_target
train_val_margin = training_config.train_val_margin
log_filename = training_config.log_filename

log_buffer = []
if master_process:
    log_buffer.extend([
        f"python: {sys.version}",
        f"torch: {torch.__version__}",
        f"torch.cuda: {torch.version.cuda}",
        f"triton: {triton.__version__}",
        "=" * 100,
    ])
    with open(sys.argv[0], "r") as f:
        log_buffer.append(f.read())
    log_buffer.append("=" * 100)
    for message in (
        f"per-gpu gradient accumulation mini-steps: {grad_accum_mini_steps}",
        f"max train steps: {max_train_steps:,}",
    ):
        print(message)
        log_buffer.append(message)

torch.set_float32_matmul_precision(training_config.float32_matmul_precision)
if master_process:
    print(
        f"{training_config.val_tokens:,} val tokens to be consumed in {val_steps:,} steps "
        f"({training_config.total_tokens_per_step_val:,} tokens per val step)"
    )
    os.makedirs(training_config.config_and_log_dir, exist_ok=True)
    if training_config.enable_checkpointing:
        os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    if training_config.export_full_model_at_end or training_config.full_model_export_interval > 0:
        os.makedirs(training_config.model_export_dir, exist_ok=True)

allow_val = False
start_step = 0
train_tokens_processed = 0
total_train_t = 0.0
total_val_t = 0.0
total_sample_t = 0.0
total_hellaswag_t = 0.0
total_t = 0.0
best_val_loss = float("inf")
best_improvement_count = 0
schedule_state = None
epoch = 0
current_shard_idx = 0
grad_accum_mini_steps_per_shard_counter = 0

seed = training_config.base_seed + rank if parallel_mode == "ddp" else training_config.base_seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

gpt_config = GPTConfig()
resume_config = {
    "model": asdict(gpt_config),
    "training": {
        name: getattr(training_config, name)
        for name in (
            "parallel_mode", "total_tokens_per_step_train", "gpu_batch_size_train", "seq_len_train",
            "grad_accum_mini_steps", "batch_size_schedule_keys", "batch_size_schedule_values",
            "swa_schedule_keys", "swa_schedule_values", "max_seq_len", "flex_block_size",
            "swa_initial_window_size",
            "data_path", "data_uses_padding", "base_seed", "optimizer_type",
            "adamw_betas", "adamw_eps", "adamw_weight_decay", "muon_lr_scale",
            "muon_backend", "muon_backend_steps", "muon_momentum", "muon_use_nesterov",
            "moe_load_balance_weight", "use_bf16_autocast", "use_all_bf16_and_null_ctx",
            "use_bf16_weights_params_or_scales",
            "keep_1d_weights_params_and_scales_in_fp32", "keep_fp32_loss",
            "cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast",
        )
    },
    "bf16_weights_params_and_scales": {
        name: f"{value.__module__}.{value.__qualname__}"
        for name, value in training_config.bf16_weights_params_and_scales.items()
    },
    "lr_schedule": [
        {"start": spec["start"], "end": spec["end"], "fn": spec["fn"].__name__, "kwargs": spec["kwargs"]}
        for spec in training_config.lr_schedule
    ],
}
if training_config.resume_from_checkpoint:
    (start_step, torch_rng_state_cpu, torch_rng_state_cuda, train_tokens_processed,
        total_train_t, total_val_t, total_sample_t, total_hellaswag_t, total_t, best_val_loss, epoch,
        current_shard_idx, grad_accum_mini_steps_per_shard_counter, schedule_state,
        best_improvement_count, allow_val, optimizer_state_dicts, gpt_model) = load_checkpoint(
            training_config, gpt_config, rank, world_size, resume_config
        )
    if master_process:
        message = f"loaded checkpoint from step {start_step - 1}"
        print(message)
        log_buffer.append(message)
    dist.barrier()
else:
    stage_index = rank if parallel_mode == "pp" else None
    gpt_model = GPT(gpt_config, training_config, stage_index=stage_index)

gpt_model.to(device)
if training_config.use_all_bf16_and_null_ctx or training_config.use_bf16_weights_params_or_scales:
    convert_to_bf16(
        gpt_model,
        training_config.bf16_weights_params_and_scales,
        training_config.keep_1d_weights_params_and_scales_in_fp32,
    )
gpt_model = torch.compile(
    gpt_model,
    mode=training_config.torch_compile_mode,
    fullgraph=training_config.torch_compile_fullgraph,
    dynamic=training_config.torch_compile_dynamic,
)
if parallel_mode == "ddp":
    gpt_model = DDP(gpt_model, device_ids=[local_rank])
    raw_gpt_model = gpt_model.module
else:
    raw_gpt_model = gpt_model

optimizers = raw_gpt_model.configure_optimizers(
    training_config.adamw_max_lr,
    training_config.adamw_betas,
    training_config.adamw_eps,
    training_config.adamw_weight_decay,
    device_type,
    muon_class=Muon,
    master_process=master_process,
    log_buffer=log_buffer,
)
if training_config.resume_from_checkpoint:
    if set(optimizers) != set(optimizer_state_dicts):
        raise ValueError("checkpoint optimizer names differ from the current run")
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(optimizer_state_dicts[name])
    dist.barrier()

pipeline_runtime = PipelineRuntime(
    gpt_model, rank, world_size, device, grad_accum_mini_steps,
    batch_size_schedule_values, raw_gpt_model.use_doc_masking,
) if parallel_mode == "pp" else None
pipeline_next_batch = pipeline_runtime.next_batch if pipeline_runtime is not None else None
pipeline_train_step = pipeline_runtime.train_step if pipeline_runtime is not None else None
pipeline_non_training_forward = pipeline_runtime.non_training_forward if pipeline_runtime is not None else None

ctx = precision_context(training_config, device_type)
loader_world_size = world_size if parallel_mode == "ddp" else 1
loader_rank = rank if parallel_mode == "ddp" else 0
use_loader = parallel_mode == "ddp" or master_process
train_data_loader = DataLoader(
    gpu_batch_size_train, seq_len_train, loader_world_size, loader_rank, training_config.data_path, "train",
    epoch=epoch, current_shard_idx=current_shard_idx,
    grad_accum_mini_steps_per_shard_counter=grad_accum_mini_steps_per_shard_counter,
    pad_token_id=gpt_config.pad_token_id, eos_token_id=gpt_config.eos_token_id,
    return_document_ids=raw_gpt_model.use_doc_masking,
    master_process=master_process, log_buffer=log_buffer,
) if use_loader else None
val_data_loader = DataLoader(
    gpu_batch_size_val, seq_len_val, loader_world_size, loader_rank, training_config.data_path, "val",
    epoch=0, current_shard_idx=0, grad_accum_mini_steps_per_shard_counter=0,
    pad_token_id=gpt_config.pad_token_id, eos_token_id=gpt_config.eos_token_id,
    return_document_ids=raw_gpt_model.use_doc_masking,
    shuffle_val_tokens=training_config.shuffle_val_tokens,
    master_process=master_process, log_buffer=log_buffer,
) if use_loader else None


def forward_for_evaluation(model, indices, targets=None, attn_mask=None, ignore_doc_mask=False, document_ids=None):
    if parallel_mode == "pp":
        return pipeline_non_training_forward(model, indices, targets, attn_mask, ignore_doc_mask, document_ids)
    return model(indices, targets, attn_mask=attn_mask, ignore_doc_mask=ignore_doc_mask, document_ids=document_ids)


run = SimpleNamespace(
    parallel_mode=parallel_mode,
    debug_nonfinite=training_config.debug_nonfinite and parallel_mode == "ddp" and world_size == 1,
    rank=rank,
    world_size=world_size,
    master_process=master_process,
    device=device,
    training_config=training_config,
    gpt_model=gpt_model,
    raw_gpt_model=raw_gpt_model,
    optimizers=optimizers,
    tokenizer=tokenizer,
    ctx=ctx,
    log_buffer=log_buffer,
    forward=forward_for_evaluation,
    pipeline_next_batch=pipeline_next_batch,
    pipeline_train_step=pipeline_train_step,
    pipeline_non_training_forward=pipeline_non_training_forward,
)
load_hellaswag_data(run)
if master_process:
    save_config_info(run, training_config, gpt_config)
log_parameter_counts(run, gpt_config)
if run.debug_nonfinite:
    message = (
        f"non-finite diagnostics enabled for single-GPU DDP | seed: {seed} | "
        f"batch sizes: {batch_size_schedule_values} | "
        f"bf16 autocast: {training_config.use_bf16_autocast} | "
        f"compile mode: {training_config.torch_compile_mode} | timings include diagnostic checks"
    )
    print(message)
    log_buffer.append(message)
    flush_log(run)
check_finite_tensors(run, raw_gpt_model.named_parameters(), "parameters before warmup")
kernel_warmup(run, num_train_steps=training_config.kernel_warmup_train_steps)
check_finite_tensors(run, raw_gpt_model.named_parameters(), "parameters after warmup restoration")
if training_config.resume_from_checkpoint:
    torch.set_rng_state(torch_rng_state_cpu)
    torch.cuda.set_rng_state(torch_rng_state_cuda)

##################################################################
#          Training, Validation, Sampling, HellaSwag loop        #
##################################################################
try:
    swa_schedule_keys = training_config.swa_schedule_keys
    swa_schedule_values = training_config.swa_schedule_values
    swa_schedule_idx = 0
    current_window_size = swa_schedule_values[swa_schedule_idx]
    next_swa_update_tokens = get_next_update_tokens(
        swa_schedule_keys,
        swa_schedule_idx,
        training_config.max_tokens
    )

    batch_schedule_idx = 0
    current_batch_size = batch_size_schedule_values[batch_schedule_idx]
    next_batch_update_tokens = get_next_update_tokens(
        batch_size_schedule_keys,
        batch_schedule_idx,
        training_config.max_tokens
    )

    lr_schedule_idx = 0
    next_lr_update_tokens = lr_schedule[lr_schedule_idx]["end"]

    if schedule_state is not None:
        swa_schedule_idx = schedule_state["swa_schedule_idx"]
        current_window_size = schedule_state["current_window_size"]
        next_swa_update_tokens = schedule_state["next_swa_update_tokens"]
        batch_schedule_idx = schedule_state["batch_schedule_idx"]
        current_batch_size = schedule_state["current_batch_size"]
        next_batch_update_tokens = schedule_state["next_batch_update_tokens"]
        lr_schedule_idx = schedule_state["lr_schedule_idx"]
        next_lr_update_tokens = schedule_state["next_lr_update_tokens"]
    raw_gpt_model.sliding_window_size.fill_(current_window_size)

    for step in range(start_step, max_train_steps):
        
        torch.cuda.synchronize()
        start_train_t = time.time()
        gpt_model.train()
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        train_loss = 0.0
        checkpoint_val_loss = None
        validation_improved = False
        stop_training = False

        # get batch size
        if train_tokens_processed >= next_batch_update_tokens:
            if batch_schedule_idx + 1 < len(batch_size_schedule_keys):
                batch_schedule_idx += 1
                current_batch_size = batch_size_schedule_values[batch_schedule_idx]
                next_batch_update_tokens = get_next_update_tokens(
                    batch_size_schedule_keys,
                    batch_schedule_idx,
                    training_config.max_tokens
                )

        if parallel_mode == "pp":
            pipeline_batches = []
            for mini_step in range(grad_accum_mini_steps):
                x_train, y_train, doc_ids_train = pipeline_next_batch(train_data_loader, current_batch_size, seq_len_train)
                pipeline_batches.append((x_train, y_train, doc_ids_train))
            with ctx:
                train_losses = pipeline_train_step(pipeline_batches)
            # with 2 GPUs, stage 0 produces loss = 0 (because it's a partial run) but is still
            # reduced so we sum stage 0's 0 loss with stage 1's actual loss
            dist.all_reduce(train_losses, op=dist.ReduceOp.SUM)
        else:
            diagnostic_context = f"step {step}, batch size {current_batch_size}"
            check_finite_tensors(run, raw_gpt_model.named_parameters(), f"{diagnostic_context}, before forward")
            train_losses = torch.zeros(3, device=device)
            for mini_step in range(grad_accum_mini_steps):
                x_train, y_train, doc_ids_train = train_data_loader.next_batch()
                x_train = x_train[:current_batch_size].pin_memory().to(device, non_blocking=True)
                y_train = y_train[:current_batch_size].pin_memory().to(device, non_blocking=True)
                if doc_ids_train is not None:
                    doc_ids_train = doc_ids_train[:current_batch_size].pin_memory().to(device, non_blocking=True)
                gpt_model.require_backward_grad_sync = (mini_step == grad_accum_mini_steps - 1)
                with ctx:
                    step_train_loss, step_token_loss, step_balance_term = gpt_model(x_train, y_train, document_ids=doc_ids_train)
                check_finite_tensors(
                    run,
                    (("train loss", step_train_loss), ("train token loss", step_token_loss), ("train balance term", step_balance_term)),
                    f"{diagnostic_context}, microbatch {mini_step + 1}/{grad_accum_mini_steps}, after forward",
                )
                train_losses += torch.stack((step_train_loss.detach().float(), step_token_loss.detach().float(), step_balance_term.detach().float())) / grad_accum_mini_steps
                (step_train_loss / grad_accum_mini_steps).backward()
                check_finite_tensors(
                    run,
                    ((name, parameter.grad) for name, parameter in raw_gpt_model.named_parameters() if parameter.grad is not None),
                    f"{diagnostic_context}, microbatch {mini_step + 1}/{grad_accum_mini_steps}, after backward",
                )
            dist.all_reduce(train_losses, op=dist.ReduceOp.AVG)
        train_loss, token_loss, balance_term = train_losses.unbind()
        
        # # if clipping is enabled and threshold > 0, use it; otherwise use +inf (no-op clip).
        # grad_clip_enabled = raw_gpt_model.use_grad_norm_clipping
        # gradient_clipping_norm = raw_gpt_model.gradient_clipping_norm if (grad_clip_enabled and raw_gpt_model.gradient_clipping_norm > 0.0) else float('inf')
        # # instead of clamping each weight’s gradient (individually), we compute the global L2
        # # norm of all gradients across the parameter set and if ||G|| > max_norm: we scale 
        # # all gradients down proportionally
        # # -------------
        # # the steps are:
        # # forward pass -> loss
        # # loss.backward() -> autograd computes true grads in p.grad
        # # (if using AMP GradScaler) scaler.unscale_(optimizer)
        # # clip grads (global norm) -> in-place scale of p.grad
        # # optimizer.step()
        # # optimizer.zero_grad() (next step)
        # # -------------
        # # Hence, after deriving everything (dJ/dw_i for each w_i), we have
        # # G which can be seen as a (total_params, 1) grad column vector 
        # # that (if gradient clipping is applied) is rescaled to 
        # # have its L2 norm <= grad_clip_norm_threshold
        # grad_norm_pre_clipping = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), gradient_clipping_norm)
        # # recompute global norm after clipping (cheap)
        # if grad_clip_enabled:
        #     # grad_sq = [p.grad.detach().float().pow(2).sum() for p in gpt_model.parameters() if p.grad is not None]
        #     # grad_norm_post_clipping = float(torch.stack(grad_sq).sum().sqrt())
        #     grad_norm_post_clipping = grad_norm_pre_clipping if grad_norm_pre_clipping < raw_gpt_model.gradient_clipping_norm else raw_gpt_model.gradient_clipping_norm
        # else:
        #     grad_norm_post_clipping = grad_norm_pre_clipping
        # grad_norm_text = f"{grad_norm_pre_clipping:.4f}" + (f" pre- / {grad_norm_post_clipping:.4f} post-clipping" if grad_clip_enabled else "")
        # <-- Uncomment for gradient norm clipping -->
        # grad_norm_pre_clipping = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(),raw_gpt_model.gradient_clipping_norm)
        # grad_norm_text = (
        #         f"{float(grad_norm_pre_clipping):.4f} pre-"
        # )
        # <-- Uncomment for gradient norm clipping -->

        # update lr
        if train_tokens_processed >= next_lr_update_tokens:
            if lr_schedule_idx + 1 < len(lr_schedule):
                lr_schedule_idx += 1
                next_lr_update_tokens = lr_schedule[lr_schedule_idx]["end"]
        lr_spec = lr_schedule[lr_schedule_idx]
        lr_tokens = train_tokens_processed - lr_spec["start"]
        adamw_lr = lr_spec["fn"](lr_tokens, **lr_spec["kwargs"])

        # update SWA window size (token schedule)
        if train_tokens_processed >= next_swa_update_tokens:
            if swa_schedule_idx + 1 < len(swa_schedule_keys):
                swa_schedule_idx += 1
                current_window_size = swa_schedule_values[swa_schedule_idx]
                next_swa_update_tokens = get_next_update_tokens(
                    swa_schedule_keys,
                    swa_schedule_idx,
                    training_config.max_tokens
                )
        # update the buffer in-place to avoid graph breaks
        raw_gpt_model.sliding_window_size.fill_(current_window_size)

        # AdamW gets the base learning rate
        for param_group in optimizers["adamw"].param_groups:
            param_group['lr'] = adamw_lr
        # Muon if present gets scaled learning rate
        if "muon" in optimizers:
            for param_group in optimizers["muon"].param_groups:
                param_group['lr'] = adamw_lr * raw_gpt_model.muon_lr_scale
        # step present optimizers
        for optimizer_name, optimizer in optimizers.items():
            optimizer.step()
            check_finite_tensors(
                run, raw_gpt_model.named_parameters(),
                f"step {step}, batch size {current_batch_size}, after {optimizer_name} update",
            )

        # the logits can vary across ranks
        # <-- Uncomment for logit soft-capping -->
        # logits_absmax_pre_capping = torch.tensor(float(raw_gpt_model._logits_absmax_stats["logits_absmax_pre_capping"]), device=device)
        # logits_absmax_post_capping = torch.tensor(float(raw_gpt_model._logits_absmax_stats["logits_absmax_post_capping"]), device=device)
        # # MAX or MEAN
        # dist.all_reduce(logits_absmax_pre_capping, op=dist.ReduceOp.MAX)
        # dist.all_reduce(logits_absmax_post_capping, op=dist.ReduceOp.MAX)
        # <-- Uncomment for logit soft-capping -->

        tl = float(train_loss)
        token_tl = float(token_loss)
        balance_tl = float(balance_term)

        torch.cuda.synchronize()
        end_train_t = time.time()
        train_step_t = end_train_t - start_train_t
        if parallel_mode == "pp":
            train_step_time = torch.tensor(train_step_t, dtype=torch.float64, device=device)
            dist.all_reduce(train_step_time, op=dist.ReduceOp.MAX)
            train_step_t = train_step_time.item()
        total_train_t += train_step_t
        total_t += train_step_t
        
        # obtain the actual tokens trained on this step (which can vary due to varying batch size)
        # on all gpus (so each updates the schedule), *not just on master*
        actual_tokens_this_step = current_batch_size * seq_len_train * effective_world_size * grad_accum_mini_steps
        train_tokens_processed += actual_tokens_this_step

        qk_suffix = ""
        if raw_gpt_model.use_qk_norm and raw_gpt_model.use_qk_debug_log:
            if parallel_mode == "pp":
                stage_qk_debug = [None for _ in range(world_size)]
                dist.all_gather_object(stage_qk_debug, qk_scale_debug_string(raw_gpt_model))
                qk_suffix = "".join(f"stage {rank}: {value}" for rank, value in enumerate(stage_qk_debug))
            elif master_process:
                qk_suffix = qk_scale_debug_string(raw_gpt_model)

        if master_process:
            # <-- Uncomment for logit soft-capping -->
            # if raw_gpt_model.use_lm_head_logit_softcapping:
            #     logits_suffix = f" | logits absmax pre/post {logits_absmax_pre_capping.item():.4f}/{logits_absmax_post_capping.item():.4f}"
            # else:
            #     logits_suffix = f" | logits absmax {logits_absmax_pre_capping.item():.4f}"
            # <-- Uncomment for logit soft-capping -->
            # scales belong to different blocks at each pipeline stage
            sw_size_suffix = f"sw size: {raw_gpt_model.sliding_window_size}"

            train_log_content = (
                f"step: {step:,} | "
                f"train loss: {tl:.8f} | "
                f"train token loss: {token_tl:.8f} | "
                f"train balance term: {balance_tl:.8f} | "
                f"train ppl: {math.exp(token_tl):,.2f} | "
                f"train step time: {1000*(train_step_t):,.2f} ms | "
                # f"grad norm: {grad_norm_text} | "
                f"adamw lr: {adamw_lr:.8f} | "
                f"tok/s: {actual_tokens_this_step / train_step_t:,.2f} | "
                f"total toks: {train_tokens_processed:,} | "
                f"total train time: {total_train_t/60:,.2f} min | "
                f"{sw_size_suffix} | "
                f"{qk_suffix}"
                f"batch size: {current_batch_size:,}"
                # f"{logits_suffix}"
            )
            print(train_log_content)
            # with open(log_filename, "a") as f:
            #     f.write(train_log_content + "\n")
            log_buffer.append(train_log_content)

        if training_config.run_sampling and ((step % sample_interval == 0 and step > 0) or step == max_train_steps - 1):
            torch.cuda.synchronize()
            start_sample_t = time.time()
            max_new_tokens = get_sample_token_count(step)
            sample(run, sample_sequences, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            end_sample_t = time.time()
            sample_step_t = end_sample_t - start_sample_t
            total_sample_t += sample_step_t
            total_t += sample_step_t
            if master_process:
                message = f"step: {step:,} | sampling time: {(sample_step_t):,.2f} s"
                print(message)
                log_buffer.append(message)

        if training_config.run_benchmarks and ((step % hellaswag_interval == 0 and step > 0) or step == max_train_steps - 1):
            torch.cuda.synchronize()
            start_hellaswag_t = time.time()
            accuracy = evaluate_hellaswag_standard(run)
            torch.cuda.synchronize()
            end_hellaswag_t = time.time()
            hellaswag_step_t = end_hellaswag_t - start_hellaswag_t
            total_hellaswag_t += hellaswag_step_t
            total_t += hellaswag_step_t
            if master_process:
                message = f"step: {step:,} | HellaSwag acc: {accuracy:.4f} | HellaSwag time: {(hellaswag_step_t):,.2f} s"
                print(message)
                log_buffer.append(message)
        
        # val gating
        if (tl + train_val_margin <= val_target) and not allow_val:
            allow_val = True
            if master_process:
                message = (f"val enabled at step {step} - {tl} train loss")
                print(message)
                log_buffer.append(message)
        
        if (allow_val and (step % val_interval == 0 and step > 0)) or step == max_train_steps - 1:
            torch.cuda.synchronize()
            start_val_t = time.time()
            if master_process:
                message = f"resetting val loader at step {step}"
                print(message)
                log_buffer.append(message)
            val_loss = evaluate_validation(run, val_data_loader, val_steps, gpu_batch_size_val, seq_len_val)
            checkpoint_val_loss = val_loss.item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_improvement_count += 1
                validation_improved = True
                if master_process:
                    message = f"new best val loss: {best_val_loss:.8f}"
                    print(message)
                    log_buffer.append(message)
                    # write buffered logs when validation improves
                    with open(log_filename, "a") as f:
                        for line in log_buffer:
                            f.write(line + "\n")
                        log_buffer.clear()
                dist.barrier()

            torch.cuda.synchronize()
            end_val_t = time.time()
            val_step_t = end_val_t - start_val_t
            total_val_t += val_step_t
            total_t += val_step_t

            if master_process:
                val_log_content = f"step: {step:,} | val loss: {val_loss.item():.8f} | val ppl: {math.exp(val_loss.item()):,.2f} | val time: {1000*(val_step_t):,.2f} ms"
                print(val_log_content)
                # with open(log_filename, "a") as f:
                #     f.write(val_log_content + "\n")
                log_buffer.append(val_log_content)

            if val_loss <= val_target:
                if master_process:
                    print(f"val loss {val_loss.item():.8f} reached target {val_target}")
                stop_training = True

        periodic_checkpoint = (
            training_config.checkpoint_interval > 0
            and (step + 1) % training_config.checkpoint_interval == 0
        )
        best_checkpoint = (
            validation_improved
            and training_config.checkpoint_best_improvement_interval > 0
            and best_improvement_count % training_config.checkpoint_best_improvement_interval == 0
        )
        checkpoint_model_path = None
        if training_config.enable_checkpointing and (periodic_checkpoint or best_checkpoint):
            loader_position = [(
                train_data_loader.epoch,
                train_data_loader.current_shard_idx,
                train_data_loader.grad_accum_mini_steps_per_shard_counter,
            ) if master_process else None]
            dist.broadcast_object_list(loader_position, src=0)
            checkpoint_rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
            if parallel_mode == "ddp":
                rng_states_by_rank = [None] * world_size if master_process else None
                dist.gather_object(checkpoint_rng, rng_states_by_rank, dst=0)
            else:
                rng_states_by_rank = [checkpoint_rng]

            if parallel_mode == "pp" or master_process:
                checkpoint_model_path = save_checkpoint(
                    training_config.checkpoint_dir, gpt_model, log_buffer,
                    step=step, rng_states_by_rank=rng_states_by_rank,
                    train_loss=train_loss.item(), val_loss=checkpoint_val_loss,
                    train_tokens_processed=train_tokens_processed,
                    total_train_t=total_train_t, total_val_t=total_val_t,
                    total_sample_t=total_sample_t, total_hellaswag_t=total_hellaswag_t,
                    total_t=total_t, best_val_loss=float(best_val_loss),
                    epoch=loader_position[0][0], current_shard_idx=loader_position[0][1],
                    grad_accum_mini_steps_per_shard_counter=loader_position[0][2],
                    optimizers=optimizers, parallel_mode=parallel_mode,
                    world_size=world_size, rank=rank, resume_config=resume_config,
                    schedule_state={
                        "swa_schedule_idx": swa_schedule_idx,
                        "current_window_size": current_window_size,
                        "next_swa_update_tokens": next_swa_update_tokens,
                        "batch_schedule_idx": batch_schedule_idx,
                        "current_batch_size": current_batch_size,
                        "next_batch_update_tokens": next_batch_update_tokens,
                        "lr_schedule_idx": lr_schedule_idx,
                        "next_lr_update_tokens": next_lr_update_tokens,
                    },
                    best_improvement_count=best_improvement_count,
                    allow_val=allow_val,
                )
            dist.barrier()
            if master_process:
                keep_latest_checkpoints(
                    training_config.checkpoint_dir,
                    training_config.max_checkpoints_to_keep,
                    log_buffer,
                )
            dist.barrier()

        export_due = (
            (training_config.export_full_model_at_end and (stop_training or step == max_train_steps - 1))
            or (training_config.full_model_export_interval > 0 and (step + 1) % training_config.full_model_export_interval == 0)
        )
        if export_due:
            if parallel_mode == "pp":
                stage_path = checkpoint_model_path
                if stage_path is None:
                    stage_path = save_stage_weights_for_export(gpt_model, training_config.model_export_dir, step, rank)
                stage_paths = [None] * world_size if master_process else None
                dist.gather_object(stage_path, stage_paths, dst=0)
                if master_process:
                    export_path = export_full_model_pp(
                        stage_paths, resume_config, training_config.model_export_dir, step, current_window_size
                    )
                dist.barrier()
                if checkpoint_model_path is None:
                    os.remove(stage_path)
                dist.barrier()
            elif master_process:
                export_path = export_full_model_ddp(
                    gpt_model, resume_config, training_config.model_export_dir, step,
                    current_window_size, checkpoint_model_path
                )
            if master_process:
                message = f"exported full model to: {export_path}"
                print(message)
                log_buffer.append(message)

        if stop_training:
            break

except Exception as e:
    if master_process:
        print(f"[rank {rank}] unhandled exception: {e}")
        import traceback
        traceback.print_exc()
    dist.barrier()
    raise
finally:
    flush_log(run)
    dist.barrier()
    destroy_process_group()
