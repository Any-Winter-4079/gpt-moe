from __future__ import annotations

from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from data.loader import DataLoader


class PipelineStageModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
            self,
            indices: Tensor,
            previous_stage_balance_loss: Optional[Tensor] = None,
            targets: Optional[Tensor] = None,
            document_ids: Optional[Tensor] = None,
            ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        output = self.model(indices, targets, document_ids=document_ids, previous_stage_balance_loss=previous_stage_balance_loss)
        # the scheduler transports activations from the first stage and backpropagates the final stage's loss
        return output


class PipelineRuntime:
    def __init__(
            self,
            model: nn.Module,
            rank: int,
            world_size: int,
            device: str,
            grad_accum_mini_steps: int,
            batch_size_schedule_values: List[int],
            use_doc_masking: bool,
            ) -> None:
        self.master_process = rank == 0
        self.device = device
        self.use_doc_masking = use_doc_masking
        self.loss_components: List[Tuple[Tensor, Tensor]] = []
        self.schedules = {
            batch_size: Schedule1F1B(
                PipelineStage(PipelineStageModel(model), rank, world_size, torch.device(device)),
                n_microbatches=grad_accum_mini_steps,
                loss_fn=self.loss_fn,
                scale_grads=True,
            )
            for batch_size in sorted(set(batch_size_schedule_values), reverse=True)
        }

    def next_batch(self, data_loader: Optional[DataLoader], batch_size: int, seq_len: int) -> Tuple[Optional[Tensor], Tensor, Optional[Tensor]]:
        if self.master_process:
            x, y, doc_ids = data_loader.next_batch()
            x = x[:batch_size].pin_memory().to(self.device, non_blocking=True)
            y = y[:batch_size].pin_memory().to(self.device, non_blocking=True)
            if doc_ids is not None:
                doc_ids = doc_ids[:batch_size].pin_memory().to(self.device, non_blocking=True)
        else:
            x = None
            y = torch.empty((batch_size, seq_len), dtype=torch.long, device=self.device)
            doc_ids = torch.empty_like(y) if self.use_doc_masking else None
        # broadcast from one loader to avoid duplicating CPU shard storage, prefetching, and batch preparation
        # targets belong to the final stage; document masks must agree at both stages
        dist.broadcast(y, src=0)
        if doc_ids is not None:
            dist.broadcast(doc_ids, src=0)
        return x, y, doc_ids

    def loss_fn(self, output: Tuple[Tensor, Tensor, Tensor, Tensor], target: Tensor) -> Tensor:
        logits, total_loss, token_loss, balance_term = output
        self.loss_components.append((token_loss.detach().float(), balance_term.detach().float()))
        return total_loss

    def train_step(self, batches: List[Tuple[Optional[Tensor], Tensor, Optional[Tensor]]]) -> Tensor:
        schedule = self.schedules[batches[0][1].size(0)]
        doc_ids = torch.cat([batch[2] for batch in batches]) if batches[0][2] is not None else None
        losses = []
        self.loss_components.clear()
        if self.master_process:
            x = torch.cat([batch[0] for batch in batches])
            schedule.step(x, document_ids=doc_ids, return_outputs=False)
        else:
            y = torch.cat([batch[1] for batch in batches])
            # targets feed GPT.forward; target is also required by the schedule's loss callback
            schedule.step(target=y, targets=y, document_ids=doc_ids, losses=losses, return_outputs=False)
        # only the final stage contributes losses to the subsequent SUM reduction
        return torch.stack([
            torch.stack((loss.detach().float(), token_loss, balance_term))
            for loss, (token_loss, balance_term) in zip(losses, self.loss_components)
        ]).mean(dim=0).float() if losses else torch.zeros(3, device=self.device)

    @torch.no_grad()
    def non_training_forward(
            self,
            model: nn.Module,
            indices: Optional[Tensor],
            targets: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            ignore_doc_mask: bool = False,
            document_ids: Optional[Tensor] = None,
            ) -> Optional[Tensor]:
        # forward-only passes retain the original batch shapes, including single prompts and variable-length HellaSwag batches
        if self.master_process:
            x = model(indices, attn_mask=attn_mask, ignore_doc_mask=ignore_doc_mask, document_ids=document_ids)
            metadata = [(x.shape, x.dtype)]
        else:
            metadata = [None]
        dist.broadcast_object_list(metadata, src=0, device=torch.device(self.device))
        if self.master_process:
            dist.send(x.contiguous(), dst=1)
            return torch.zeros((), device=self.device) if targets is not None else None
        shape, dtype = metadata[0]
        x = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(x, src=0)
        return model(x, targets, attn_mask=attn_mask, ignore_doc_mask=ignore_doc_mask, document_ids=document_ids)
