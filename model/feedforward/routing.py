from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


@triton.jit
def _count_kernel(
    Indices, Histogram,
    M: tl.constexpr, NUM_EXPERTS: tl.constexpr, NUM_CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr, EXPERT_BINS: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = chunk * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(Indices + row, mask=row < M, other=0).to(tl.int32)
    count = tl.histogram(expert, EXPERT_BINS, mask=row < M)
    expert_id = tl.arange(0, EXPERT_BINS)
    tl.store(Histogram + expert_id * NUM_CHUNKS + chunk, count, mask=expert_id < NUM_EXPERTS)


@triton.jit
def _prefix_kernel(
    Histogram, ChunkStarts, Offsets,
    NUM_EXPERTS: tl.constexpr, NUM_CHUNKS: tl.constexpr,
    EXPERT_BINS: tl.constexpr, CHUNK_BINS: tl.constexpr,
):
    expert = tl.arange(0, EXPERT_BINS)
    chunk = tl.arange(0, CHUNK_BINS)
    valid = (expert[:, None] < NUM_EXPERTS) & (chunk[None, :] < NUM_CHUNKS)
    count = tl.load(Histogram + expert[:, None] * NUM_CHUNKS + chunk[None, :], mask=valid, other=0)
    earlier_chunks = tl.cumsum(count, axis=1) - count
    total = tl.sum(count, axis=1)
    end = tl.cumsum(total, axis=0)
    start = end - total
    tl.store(
        ChunkStarts + expert[:, None] * NUM_CHUNKS + chunk[None, :],
        start[:, None] + earlier_chunks, mask=valid,
    )
    tl.store(Offsets + expert, end, mask=expert < NUM_EXPERTS)


@triton.jit
def _permutation_kernel(
    Indices, ChunkStarts, ExpertIndices, Order, InverseOrder,
    M: tl.constexpr, NUM_CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr, EXPERT_BINS: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = chunk * BLOCK + tl.arange(0, BLOCK)
    selected = tl.load(Indices + row, mask=row < M, other=0).to(tl.int32)
    expert = tl.arange(0, EXPERT_BINS)
    belongs = (expert[:, None] == selected[None, :]) & (row[None, :] < M)
    local_rank = tl.cumsum(belongs.to(tl.int32), axis=1)
    rank = tl.sum(tl.where(belongs, local_rank, 0), axis=0)
    start = tl.load(ChunkStarts + selected * NUM_CHUNKS + chunk)
    destination = start + rank - 1
    # preserve original token/top-k slot order within each expert without atomics
    tl.store(ExpertIndices + destination, selected, mask=row < M)
    tl.store(Order + destination, row, mask=row < M)
    tl.store(InverseOrder + row, destination, mask=row < M)


@triton_op("gpt_moe::counted_expert_routing", mutates_args=())
def counted_expert_routing(indices: Tensor, num_experts: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    indices = indices.reshape(-1)
    m = indices.numel()
    expert_indices = torch.empty_like(indices)
    order = torch.empty_like(indices)
    inverse_order = torch.empty_like(indices)
    offsets = torch.empty((num_experts,), device=indices.device, dtype=torch.int32)
    if m == 0:
        offsets.zero_()
        return expert_indices, order, inverse_order, offsets

    block = 512
    num_chunks = triton.cdiv(m, block)
    expert_bins = triton.next_power_of_2(num_experts)
    histogram = torch.empty((num_experts, num_chunks), device=indices.device, dtype=torch.int32)
    chunk_starts = torch.empty_like(histogram)
    wrap_triton(_count_kernel)[(num_chunks,)](
        indices, histogram, m, num_experts, num_chunks, block, expert_bins, num_warps=4,
    )
    wrap_triton(_prefix_kernel)[(1,)](
        histogram, chunk_starts, offsets, num_experts, num_chunks,
        expert_bins, triton.next_power_of_2(num_chunks), num_warps=4,
    )
    wrap_triton(_permutation_kernel)[(num_chunks,)](
        indices, chunk_starts, expert_indices, order, inverse_order,
        m, num_chunks, block, expert_bins, num_warps=4,
    )
    return expert_indices, order, inverse_order, offsets
