from __future__ import annotations

from typing import Any, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_stages=3, num_warps=4)
        for bm, bn, bk in [(64, 64, 64), (64, 128, 32), (64, 128, 64), (128, 64, 32)]
    ],
    key=["M", "N", "K", "NUM_EXPERTS", "GROUP_K", "a_stride_m", "a_stride_k", "b_stride_k", "b_stride_n"],
)
@triton.jit
def _grouped_mm_kernel(
    A, B, C, Offsets,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    a_stride_m: tl.constexpr, a_stride_k: tl.constexpr,
    b_stride_e: tl.constexpr, b_stride_k: tl.constexpr, b_stride_n: tl.constexpr,
    NUM_EXPERTS: tl.constexpr, NUM_PROGRAMS: tl.constexpr, GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # persistent workers traverse the tiles of all experts
    tile = tl.program_id(0)
    first_tile = 0
    start = 0
    for expert in range(NUM_EXPERTS):
        end = tl.load(Offsets + expert)
        if GROUP_K:
            rows, reduction = M, end - start
            a_base = start * a_stride_k
            b_base = start * b_stride_k
            c_base = expert * M * N
        else:
            rows, reduction = end - start, K
            a_base = start * a_stride_m
            b_base = expert * b_stride_e
            c_base = start * N

        tiles_n = tl.cdiv(N, BLOCK_N)
        last_tile = first_tile + tl.cdiv(rows, BLOCK_M) * tiles_n
        while tile < last_tile:
            local_tile = tile - first_tile
            row = (local_tile // tiles_n) * BLOCK_M + tl.arange(0, BLOCK_M)
            col = (local_tile % tiles_n) * BLOCK_N + tl.arange(0, BLOCK_N)
            inner = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for block in range(tl.cdiv(reduction, BLOCK_K)):
                k = block * BLOCK_K + inner
                a = tl.load(
                    A + a_base + row[:, None] * a_stride_m + k[None, :] * a_stride_k,
                    mask=(row[:, None] < rows) & (k[None, :] < reduction), other=0.0,
                )
                b = tl.load(
                    B + b_base + k[:, None] * b_stride_k + col[None, :] * b_stride_n,
                    mask=(k[:, None] < reduction) & (col[None, :] < N), other=0.0,
                )
                acc = tl.dot(a, b, acc, input_precision="tf32x3")
            tl.store(
                C + c_base + row[:, None] * N + col[None, :], acc.to(C.dtype.element_ty),
                mask=(row[:, None] < rows) & (col[None, :] < N),
            )
            tile += NUM_PROGRAMS
        first_tile = last_tile
        start = end


def _grouped_mm(a: Tensor, b: Tensor, offsets: Tensor) -> Tensor:
    # 2d @ 3d partitions output rows; 2d @ 2d partitions the weight-gradient reduction
    group_k = b.ndim == 2
    m, k = a.shape
    n = b.shape[-1]
    num_experts = offsets.numel()
    out = a.new_empty((num_experts, m, n) if group_k else (m, n))
    num_programs = torch.cuda.get_device_properties(a.device).multi_processor_count
    wrap_triton(_grouped_mm_kernel)[(num_programs,)](
        a, b, out, offsets, m, n, k,
        a.stride(0), a.stride(1),
        0 if group_k else b.stride(0), b.stride(-2), b.stride(-1),
        num_experts, num_programs, group_k,
    )
    return out


@triton_op("gpt_moe::grouped_linear", mutates_args=())
def grouped_linear(x: Tensor, weight: Tensor, offsets: Tensor) -> Tensor:
    return _grouped_mm(x, weight.transpose(-2, -1), offsets)


@triton_op("gpt_moe::grouped_linear_backward", mutates_args=())
def _grouped_linear_backward(grad: Tensor, x: Tensor, weight: Tensor, offsets: Tensor) -> Tuple[Tensor, Tensor]:
    grad_x = _grouped_mm(grad, weight, offsets)
    grad_weight = _grouped_mm(grad.t(), x, offsets)
    return grad_x, grad_weight


def _setup_context(ctx: Any, inputs: Tuple, output: Tensor) -> None:
    ctx.save_for_backward(*inputs)


def _backward(ctx: Any, grad: Tensor) -> Tuple:
    x, weight, offsets = ctx.saved_tensors
    grad_x, grad_weight = _grouped_linear_backward(grad, x, weight, offsets)
    return grad_x, grad_weight, None


grouped_linear.register_autograd(_backward, setup_context=_setup_context)
