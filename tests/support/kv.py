"""Test-only KV gather helper (previously KVCache.gather)."""

from __future__ import annotations

import torch


def gather(layer, block_table, context_len):
    table = torch.as_tensor(block_table, dtype=torch.long, device=layer.keys.device)
    if (
        table.ndim != 1
        or context_len < 0
        or context_len > table.numel() * layer.block_size
    ):
        raise ValueError("invalid context length or block table")
    count = (context_len + layer.block_size - 1) // layer.block_size
    used = table[:count]
    if torch.any(used < 0) or torch.any(used >= layer.keys.shape[0]):
        raise ValueError("block outside cache")
    shape = (-1, *layer.keys.shape[2:])
    return (
        layer.keys[used].reshape(shape)[:context_len],
        layer.values[used].reshape(shape)[:context_len],
    )
