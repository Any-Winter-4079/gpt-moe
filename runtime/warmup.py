from __future__ import annotations

import copy
import time

import torch
import torch.distributed as dist


def _kernel_warmup_pp(run, num_train_steps: int = 2) -> None:
    raw_gpt_model = run.raw_gpt_model
    gpt_model = run.gpt_model
    optimizers = run.optimizers
    batch_size_schedule_values = run.training_config.batch_size_schedule_values
    rank = run.rank
    seq_len_train = run.training_config.seq_len_train
    seq_len_val = run.training_config.seq_len_val
    gpu_batch_size_val = run.training_config.gpu_batch_size_val
    grad_accum_mini_steps = run.training_config.grad_accum_mini_steps
    device = run.device
    ctx = run.ctx
    activation_ctx = run.activation_ctx
    pipeline_train_step = run.pipeline_train_step
    pipeline_non_training_forward = run.pipeline_non_training_forward
    # snapshot everything so we don't "cheat"
    model_state = copy.deepcopy(raw_gpt_model.state_dict())
    optimizer_states = {k: copy.deepcopy(opt.state_dict()) for k, opt in optimizers.items()}
    rng_state_cpu = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state()

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # calculate all unique batch sizes to compile before training starts
    warmup_batch_sizes = sorted(set(batch_size_schedule_values), reverse=True)
    print(f"kernel_warmup: gpu {rank} will compile kernels for batch sizes: {warmup_batch_sizes}")

    # train-shape warmup (compile this pipeline stage)
    gpt_model.train()
    with torch.enable_grad():
        for batch_size in warmup_batch_sizes:
            print(f"kernel_warmup: gpu {rank} compiling kernels for batch size: {batch_size}")
            for _ in range(num_train_steps):
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                pipeline_batches = []
                for mini_step in range(grad_accum_mini_steps):
                    # make shapes match training
                    x_train = torch.randint(
                        0, raw_gpt_model.pad_token_id,
                        (batch_size, seq_len_train), device=device
                    )
                    y_train = torch.randint(
                        0, raw_gpt_model.pad_token_id,
                        (batch_size, seq_len_train), device=device
                    )

                    # synthesize doc_ids so the doc-masking + SWA FlexAttention path compiles
                    if raw_gpt_model.use_doc_masking:
                        # set random-ish EOS boundaries
                        step = max(16, seq_len_train // 8)
                        idxs = torch.arange(seq_len_train, device=device)[None, :]
                        rand_offsets = torch.randint(0, step, (batch_size, 1), device=device)
                        is_eos = ((idxs + rand_offsets) % step == 0)
                        doc_ids_train = torch.cumsum(is_eos.to(torch.int32), dim=1) - is_eos.to(torch.int32)
                    else:
                        doc_ids_train = None

                    dist.broadcast(y_train, src=0)
                    if doc_ids_train is not None:
                        dist.broadcast(doc_ids_train, src=0)
                    pipeline_batches.append((x_train, y_train, doc_ids_train))

                with ctx, activation_ctx:
                    pipeline_train_step(pipeline_batches)

                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # val-shape warmup (compile eval forward)
    gpt_model.eval()
    with torch.inference_mode(), ctx:
        x_val = torch.randint(
            0, raw_gpt_model.pad_token_id,
            (gpu_batch_size_val, seq_len_val), device=device
        )
        y_val = torch.randint(
            0, raw_gpt_model.pad_token_id,
            (gpu_batch_size_val, seq_len_val), device=device
        )
        if raw_gpt_model.use_doc_masking:
            step = max(16, seq_len_val // 8)
            idxs = torch.arange(seq_len_val, device=device)[None, :]
            rand_offsets = torch.randint(0, step, (gpu_batch_size_val, 1), device=device)
            is_eos = ((idxs + rand_offsets) % step == 0)
            doc_ids_val = torch.cumsum(is_eos.to(torch.int32), dim=1) - is_eos.to(torch.int32)
        else:
            doc_ids_val = None

        dist.broadcast(y_val, src=0)
        if doc_ids_val is not None:
            dist.broadcast(doc_ids_val, src=0)
        _ = pipeline_non_training_forward(gpt_model, x_val, y_val, document_ids=doc_ids_val)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # sampling/hellaswag warmup is skipped because both run via _orig_mod eager path

    # restore state
    raw_gpt_model.load_state_dict(model_state)
    for k, optimizer in optimizers.items():
        optimizer.load_state_dict(optimizer_states[k])
    torch.set_rng_state(rng_state_cpu)
    torch.cuda.set_rng_state(rng_state_cuda)

    torch.cuda.synchronize()
    dist.barrier()


def _kernel_warmup_ddp(run, num_train_steps: int = 2) -> None:
    raw_gpt_model = run.raw_gpt_model
    gpt_model = run.gpt_model
    optimizers = run.optimizers
    batch_size_schedule_values = run.training_config.batch_size_schedule_values
    rank = run.rank
    seq_len_train = run.training_config.seq_len_train
    seq_len_val = run.training_config.seq_len_val
    gpu_batch_size_val = run.training_config.gpu_batch_size_val
    grad_accum_mini_steps = run.training_config.grad_accum_mini_steps
    device = run.device
    ctx = run.ctx
    activation_ctx = run.activation_ctx
    # snapshot everything so we don't "cheat"
    model_state = copy.deepcopy(raw_gpt_model.state_dict())
    optimizer_states = {k: copy.deepcopy(opt.state_dict()) for k, opt in optimizers.items()}
    rng_state_cpu = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state()

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # calculate all unique batch sizes to compile before training starts
    warmup_batch_sizes = sorted(set(batch_size_schedule_values), reverse=True)
    print(f"kernel_warmup: gpu {rank} will compile kernels for batch sizes: {warmup_batch_sizes}")

    # train-shape warmup (compile both DDP graphs)
    gpt_model.train()
    with torch.enable_grad():
        for batch_size in warmup_batch_sizes:
            print(f"kernel_warmup: gpu {rank} compiling kernels for batch size: {batch_size}")
            for _ in range(num_train_steps):
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                for mini_step in range(grad_accum_mini_steps):
                    # mimic the real training loop’s DDP behavior
                    gpt_model.require_backward_grad_sync = (mini_step == grad_accum_mini_steps - 1)

                    # make shapes match training
                    x_train = torch.randint(
                        0, raw_gpt_model.pad_token_id,
                        (batch_size, seq_len_train), device=device
                    )
                    y_train = torch.randint(
                        0, raw_gpt_model.pad_token_id,
                        (batch_size, seq_len_train), device=device
                    )

                    # synthesize doc_ids so the doc-masking + SWA FlexAttention path compiles
                    if raw_gpt_model.use_doc_masking:
                        # set random-ish EOS boundaries
                        step = max(16, seq_len_train // 8)
                        idxs = torch.arange(seq_len_train, device=device)[None, :]
                        rand_offsets = torch.randint(0, step, (batch_size, 1), device=device)
                        is_eos = ((idxs + rand_offsets) % step == 0)
                        doc_ids_train = torch.cumsum(is_eos.to(torch.int32), dim=1) - is_eos.to(torch.int32)
                    else:
                        doc_ids_train = None

                    with ctx, activation_ctx:
                        warm_loss, _, _ = gpt_model(x_train, y_train, document_ids=doc_ids_train)

                    (warm_loss / grad_accum_mini_steps).backward()

                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # val-shape warmup (compile eval forward)
    gpt_model.eval()
    with torch.inference_mode(), ctx:
        x_val = torch.randint(
            0, raw_gpt_model.pad_token_id,
            (gpu_batch_size_val, seq_len_val), device=device
        )
        y_val = torch.randint(
            0, raw_gpt_model.pad_token_id,
            (gpu_batch_size_val, seq_len_val), device=device
        )
        if raw_gpt_model.use_doc_masking:
            step = max(16, seq_len_val // 8)
            idxs = torch.arange(seq_len_val, device=device)[None, :]
            rand_offsets = torch.randint(0, step, (gpu_batch_size_val, 1), device=device)
            is_eos = ((idxs + rand_offsets) % step == 0)
            doc_ids_val = torch.cumsum(is_eos.to(torch.int32), dim=1) - is_eos.to(torch.int32)
        else:
            doc_ids_val = None

        _ = gpt_model(x_val, y_val, document_ids=doc_ids_val)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # sampling/hellaswag warmup is skipped because both run via _orig_mod eager path

    # restore state
    raw_gpt_model.load_state_dict(model_state)
    for k, optimizer in optimizers.items():
        optimizer.load_state_dict(optimizer_states[k])
    torch.set_rng_state(rng_state_cpu)
    torch.cuda.set_rng_state(rng_state_cuda)

    torch.cuda.synchronize()
    dist.barrier()


def kernel_warmup(run, num_train_steps: int = 2) -> None:
    dist.barrier()
    torch.cuda.synchronize()
    start_t = time.perf_counter()
    if run.parallel_mode == "pp":
        _kernel_warmup_pp(run, num_train_steps)
    else:
        _kernel_warmup_ddp(run, num_train_steps)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = torch.tensor([time.perf_counter() - start_t], dtype=torch.float64, device=run.device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if run.master_process:
        seconds = float(elapsed.item())
        message = f"kernel warmup compile time: {seconds:.2f}s ({seconds / 60.0:.2f} min)"
        print(message)
        run.log_buffer.append(message)
