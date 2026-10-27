"""Runner-owned KV storage and zero-copy, per-layer views."""

from dataclasses import dataclass

import torch

from microinfer.worker.triton_kv_store import HAS_TRITON, store_kv


@dataclass(frozen=True)
class LayerKVCache:
    """Only this layer's [blocks, B, Hkv, D] tensors; no pool or layer ID."""

    keys: torch.Tensor
    values: torch.Tensor

    def __post_init__(self):
        if self.keys.ndim != 4 or self.keys.shape != self.values.shape:
            raise ValueError("K/V must have matching [blocks, B, Hkv, D] shapes")
        if (
            self.keys.dtype != self.values.dtype
            or self.keys.device != self.values.device
        ):
            raise ValueError("K/V must have matching dtype and device")

    @property
    def block_size(self):
        return self.keys.shape[1]

    def write_checked(self, slots, keys, values):
        """Checked write: -1 skips padding; validates before changing storage."""
        if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64):
            raise ValueError("slots must be a 1-D integer tensor")
        expected = (slots.numel(), *self.keys.shape[2:])
        if keys.shape != expected or values.shape != expected:
            raise ValueError("K/V shape must match slots and cache heads")
        if any(t.device != self.keys.device for t in (slots, keys, values)):
            raise ValueError("slots and K/V must be on the cache device")
        capacity = self.keys.shape[0] * self.block_size
        if torch.any(slots < -1) or torch.any(slots >= capacity):
            raise ValueError("slot outside cache")
        valid = slots >= 0
        indices = slots[valid].long()
        if indices.unique().numel() != indices.numel():
            raise ValueError("duplicate slot")
        # Convert both inputs before either write; padding-only/empty writes work.
        k = keys[valid].to(self.keys.dtype)
        v = values[valid].to(self.values.dtype)
        blocks, offsets = indices // self.block_size, indices % self.block_size
        self.write_fast(blocks, offsets, k, v)

    def write_fast(self, blocks, offsets, keys, values):
        """Hot path: caller guarantees valid, unique slots and matching dtype.

        ``blocks``/``offsets`` are precomputed once per step in
        ``AttentionMetadata.slot_indices`` and reused by every layer.
        """
        if (
            HAS_TRITON
            and self.keys.is_cuda
            and keys.is_cuda
            and keys.dtype == self.keys.dtype
            and keys.stride(-1) == 1
            and keys.stride(-2) == keys.shape[-1]
            and values.stride(-1) == 1
            and values.stride(-2) == values.shape[-1]
        ):
            store_kv(
                self.keys,
                self.values,
                blocks,
                offsets,
                keys,
                values,
                self.block_size,
            )
            return
        self.keys[blocks, offsets] = keys
        self.values[blocks, offsets] = values


class KVCache:
    def __init__(
        self,
        num_layers,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        *,
        dtype=torch.float32,
        device="cpu",
    ):
        dimensions = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        if any(d <= 0 for d in dimensions):
            raise ValueError("cache dimensions must be positive")
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.keys = torch.zeros(dimensions, dtype=dtype, device=device)
        self.values = torch.zeros_like(self.keys)

    @property
    def num_bytes(self):
        return (
            self.keys.numel() * self.keys.element_size()
            + self.values.numel() * self.values.element_size()
        )

    def layer(self, index: int) -> LayerKVCache:
        """Basic indexing shares storage; binding does not copy the cache."""
        if not 0 <= index < self.num_layers:
            raise ValueError("layer outside cache")
        return LayerKVCache(self.keys[index], self.values[index])
