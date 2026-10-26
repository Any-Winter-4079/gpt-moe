from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def prepare_polar_kernel(
    Gradients, Buffers, Packed,
    ROWS: tl.constexpr, COLS: tl.constexpr,
    grad_strides_r: tl.constexpr, grad_strides_c: tl.constexpr,
    buffer_strides_r: tl.constexpr, buffer_strides_c: tl.constexpr,
    momentum, NESTEROV: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    matrix = tl.program_id(1)
    rows, cols = offsets // COLS, offsets % COLS
    mask = offsets < ROWS * COLS
    for index in tl.static_range(len(Gradients)):
        if matrix == index:
            grad_ptrs = Gradients[index] + rows * grad_strides_r[index] + cols * grad_strides_c[index]
            buffer_ptrs = Buffers[index] + rows * buffer_strides_r[index] + cols * buffer_strides_c[index]
            gradient = tl.load(grad_ptrs, mask=mask, other=0.0).to(tl.float32)
            buffer = tl.load(buffer_ptrs, mask=mask, other=0.0).to(tl.float32)
            # retain the separate FP32 rounding of buf.mul_(momentum).add_(g)
            buffer = buffer * momentum
            buffer = buffer + gradient
            tl.store(buffer_ptrs, buffer, mask=mask)
            if NESTEROV:
                gradient = tl.fma(momentum, buffer, gradient)
                tl.store(grad_ptrs, gradient, mask=mask)
            else:
                gradient = buffer
            tl.store(Packed + matrix * ROWS * COLS + offsets, gradient.to(tl.bfloat16), mask=mask)


def prepare_polar(gradients, buffers, momentum: float, nesterov: bool):
    rows, cols = gradients[0].shape
    packed = torch.empty((len(gradients), rows, cols), dtype=torch.bfloat16, device=gradients[0].device)
    prepare_polar_kernel[(triton.cdiv(rows * cols, 1024), len(gradients))](
        gradients, buffers, packed, rows, cols,
        tuple(g.stride(0) for g in gradients), tuple(g.stride(1) for g in gradients),
        tuple(buf.stride(0) for buf in buffers), tuple(buf.stride(1) for buf in buffers),
        momentum, nesterov, BLOCK=1024, num_warps=4, enable_fp_fusion=False,
    )
    return packed
