from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.checkpoint import checkpoint


class _SavedActivation:
    def __init__(self, tensor: Tensor, stream: torch.cuda.Stream) -> None:
        self.device = tensor.device
        self.stream = stream
        self.source = tensor.detach()
        self.restored = None
        self.cpu = torch.empty_strided(
            tensor.size(), tensor.stride(), dtype=tensor.dtype, device="cpu", pin_memory=True,
        )
        self.offloaded = torch.cuda.Event()
        self.reloaded = torch.cuda.Event()
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self.cpu.copy_(self.source, non_blocking=True)
            self.offloaded.record(stream)

    def prefetch(self) -> None:
        if self.restored is not None:
            return
        # allocate on the compute stream so released backward storage can be reused
        self.restored = torch.empty_strided(
            self.cpu.size(), self.cpu.stride(), dtype=self.cpu.dtype, device=self.device,
        )
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            self.restored.copy_(self.cpu, non_blocking=True)
            self.reloaded.record(self.stream)

    def unpack(self) -> Tensor:
        self.prefetch()
        compute_stream = torch.cuda.current_stream(self.device)
        compute_stream.wait_event(self.reloaded)
        self.restored.record_stream(compute_stream)
        return self.restored


class _ActivationGroup:
    def __init__(self, stream: torch.cuda.Stream, parameter_storage: set[int], min_bytes: int) -> None:
        self.stream = stream
        self.parameter_storage = parameter_storage
        self.min_bytes = min_bytes
        self.saved = {}

    def pack(self, tensor: Tensor):
        # keep parameters, their views, small tensors and specialized layouts on the GPU
        if (type(tensor) is not Tensor or tensor.device.type != "cuda"
                or tensor.layout != torch.strided or not tensor.is_contiguous()
                or tensor.numel() * tensor.element_size() < self.min_bytes):
            return tensor.detach()
        storage = tensor.untyped_storage().data_ptr()
        if storage in self.parameter_storage:
            return tensor.detach()
        key = (storage, tensor.storage_offset(), tensor.size(), tensor.stride(), tensor.dtype)
        if key not in self.saved:
            self.saved[key] = _SavedActivation(tensor, self.stream)
        return self.saved[key]

    @staticmethod
    def unpack(value) -> Tensor:
        return value.unpack() if isinstance(value, _SavedActivation) else value

    def release_forward(self) -> None:
        if self.saved:
            last = next(reversed(self.saved.values()))
            compute_stream = torch.cuda.current_stream(last.device)
            compute_stream.wait_event(last.offloaded)
            for value in self.saved.values():
                if value.source is not None:
                    value.source.record_stream(compute_stream)
                value.source = None

    def prefetch(self) -> None:
        for value in self.saved.values():
            value.prefetch()

    def release_backward(self) -> None:
        for value in self.saved.values():
            if value.restored is not None:
                compute_stream = torch.cuda.current_stream(value.device)
                compute_stream.wait_event(value.reloaded)
                value.restored.record_stream(compute_stream)
            value.restored = None


class _OffloadState:
    def __init__(self, groups: list[_ActivationGroup]) -> None:
        self.groups = groups
        self.callback_queued = False

    def before_backward(self, index: int) -> None:
        if not self.callback_queued:
            torch.autograd.Variable._execution_engine.queue_callback(self.finish_backward)
            self.callback_queued = True
        if index + 1 < len(self.groups):
            self.groups[index + 1].release_backward()
        if index < len(self.groups):
            self.groups[index].prefetch()
        if 0 <= index - 1 < len(self.groups):
            self.groups[index - 1].prefetch()

    def finish_backward(self) -> None:
        for group in self.groups:
            group.release_backward()
            group.saved.clear()
        self.callback_queued = False


class _BackwardBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, balance_loss, state, index):
        ctx.state = state
        ctx.index = index
        ctx.set_materialize_grads(False)
        return x, balance_loss

    @staticmethod
    def backward(ctx, grad_x, grad_balance_loss):
        # wait for both output gradients before entering a block's backward
        ctx.state.before_backward(ctx.index)
        return grad_x, grad_balance_loss, None, None


def _run_block(
        block: nn.Module,
        x: Tensor,
        flex_mask: Optional[BlockMask],
        sdpa_mask: Optional[Tensor],
        use_checkpointing: bool,
        ) -> Tuple[Tensor, Optional[Tensor]]:
    if use_checkpointing:
        return checkpoint(
            block, x, flex_attn_block_mask=flex_mask, sdpa_attn_mask=sdpa_mask,
            use_reentrant=False,
        )
    return block(x, flex_attn_block_mask=flex_mask, sdpa_attn_mask=sdpa_mask)


class ActivationOffloading:
    def __init__(self, num_layers: int, min_bytes: int, use_checkpointing: bool) -> None:
        self.num_layers = num_layers
        self.min_bytes = min_bytes
        self.use_checkpointing = use_checkpointing
        self.stream = None
        self.block_forward = _run_block
        self.last_offloaded_bytes = 0

    def compile(self, **kwargs) -> None:
        self.block_forward = torch.compile(_run_block, **kwargs)

    @torch.compiler.disable
    def run(
            self,
            blocks: nn.ModuleList,
            full_attention: list[bool],
            x: Tensor,
            swa_mask: Optional[BlockMask],
            full_mask: Optional[BlockMask],
            sdpa_mask: Optional[Tensor],
            ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.stream is None and x.device.type == "cuda":
            self.stream = torch.cuda.Stream(device=x.device)
        parameter_storage = {p.untyped_storage().data_ptr() for p in blocks.parameters()}
        groups = [
            _ActivationGroup(self.stream, parameter_storage, self.min_bytes)
            for _ in range(self.num_layers)
        ]
        state = _OffloadState(groups)
        balance_loss = None
        try:
            for index, (block, is_full) in enumerate(zip(blocks, full_attention)):
                # allow one subsequent block to compute while the preceding block is copied
                if 0 <= index - 2 < len(groups):
                    groups[index - 2].release_forward()
                context = (
                    torch.autograd.graph.saved_tensors_hooks(groups[index].pack, groups[index].unpack)
                    if index < len(groups) else nullcontext()
                )
                with context:
                    x, block_balance_loss = self.block_forward(
                        block, x, full_mask if is_full else swa_mask, sdpa_mask, self.use_checkpointing,
                    )
                x, block_balance_loss = _BackwardBoundary.apply(x, block_balance_loss, state, index)
                if block_balance_loss is not None:
                    balance_loss = block_balance_loss if balance_loss is None else balance_loss + block_balance_loss
        finally:
            for group in groups:
                group.release_forward()
        self.last_offloaded_bytes = sum(
            value.cpu.numel() * value.cpu.element_size()
            for group in groups for value in group.saved.values()
        )
        return x, balance_loss
