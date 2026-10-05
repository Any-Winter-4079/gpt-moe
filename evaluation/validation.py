from __future__ import annotations

import torch
import torch.distributed as dist


def evaluate_validation(run, val_data_loader, val_steps: int, gpu_batch_size_val: int, seq_len_val: int) -> torch.Tensor:
    run.gpt_model.eval()
    if run.parallel_mode == "pp":
        if run.master_process:
            val_data_loader.reset()
    else:
        val_data_loader.reset()

    with torch.inference_mode():
        val_loss = torch.zeros((), device=run.device)
        for _ in range(val_steps):
            if run.parallel_mode == "pp":
                x_val, y_val, doc_ids_val = run.pipeline_next_batch(val_data_loader, gpu_batch_size_val, seq_len_val)
                step_val_loss = run.pipeline_non_training_forward(run.gpt_model, x_val, y_val, document_ids=doc_ids_val)
            else:
                x_val, y_val, doc_ids_val = val_data_loader.next_batch()
                x_val = x_val.pin_memory().to(run.device, non_blocking=True)
                y_val = y_val.pin_memory().to(run.device, non_blocking=True)
                if doc_ids_val is not None:
                    doc_ids_val = doc_ids_val.pin_memory().to(run.device, non_blocking=True)
                step_val_loss = run.gpt_model(x_val, y_val, document_ids=doc_ids_val)
            val_loss += step_val_loss.float() / val_steps
        reduction = dist.ReduceOp.SUM if run.parallel_mode == "pp" else dist.ReduceOp.AVG
        dist.all_reduce(val_loss, op=reduction)
    return val_loss
