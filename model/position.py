from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class Rotary(nn.Module):
    def __init__(self, head_size: int, base_theta: int, max_seq_len: int) -> None:
        super().__init__()
        assert head_size % 2 == 0, "RoPE needs even head_size"
        # head_size/2,
        # frequencies decay geometrically down to ~1/base_theta with ratio 1/base_theta^(2/head_size)
        inv_freq = 1.0 / (base_theta ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        # register_buffer registers as a non-trainable buffer that moves/casts with the model (to device/dtype)
        # persistent=False keeps this buffer out of state_dict
        # buffers are broadcast to all ranks (broadcast_buffers=True), so this stays in sync
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        # max_seq_len, head_size/2
        freqs = torch.outer(t, self.inv_freq)
        # 1, 1, max_seq_len, head_size/2
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def get_cos_sin(
            self,
            seq_len: int,
            dtype: torch.dtype, 
            device: Union[str, torch.device]
            ) -> Tuple[Tensor, Tensor]:
        # slice to current length and cast on the fly to avoid
        # shape mutation of buffers
        cos = self.cos_cached[..., :seq_len, :].to(device=device, dtype=dtype)
        sin = self.sin_cached[..., :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin


def apply_rotation(
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
        ) -> Tuple[Tensor, Tensor]:
    gpu_batch_size, n_heads, seq_len, head_size = q.shape
    _, n_kv_heads_k, _, _ = k.shape

    # gpu_batch_size, n_heads, seq_len, head_size // 2, 2
    q_pairs = q.contiguous().reshape(gpu_batch_size, n_heads, seq_len, head_size // 2, 2)
    k_pairs = k.contiguous().reshape(gpu_batch_size, n_kv_heads_k, seq_len, head_size // 2, 2)

    # gpu_batch_size, n_heads, seq_len, head_size/2 each
    q_x1, q_x2 = q_pairs[..., 0], q_pairs[..., 1]
    k_x1, k_x2 = k_pairs[..., 0], k_pairs[..., 1]

    # gpu_batch_size, n_heads, seq_len, head_size
    q_rot = torch.stack([q_x1 * cos - q_x2 * sin, q_x1 * sin + q_x2 * cos], dim=-1)
    k_rot = torch.stack([k_x1 * cos - k_x2 * sin, k_x1 * sin + k_x2 * cos], dim=-1)

    q = q_rot.reshape(gpu_batch_size, n_heads, seq_len, head_size)
    k = k_rot.reshape(gpu_batch_size, n_kv_heads_k, seq_len, head_size)
    
    return q, k
