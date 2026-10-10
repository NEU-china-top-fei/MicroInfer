"""M03 exercise: actual KV tensor storage, separate from block ownership."""

import torch


class KVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype=torch.float32,
        device="cpu",
    ):
        """Allocate keys/values [layers, blocks, B, KV_heads, head_dim].

        All dimensions must be positive. Unwritten slots are unspecified;
        num_bytes counts K and V. The scheduler, not this class, owns refcounts.
        """
        raise NotImplementedError("M03: KVCache allocation")

    @property
    def num_bytes(self) -> int:
        raise NotImplementedError("M03: KVCache.num_bytes")

    def write(
        self, layer: int, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Write [T,H,D] to flattened physical slots; -1 skips padding.

        Validate the entire call before mutation, including duplicate nonnegative
        slots, bounds and tensor shapes. Reject invalid calls with ValueError.
        """
        raise NotImplementedError("M03: KVCache.write")

    def gather(self, layer: int, block_table: list[int], context_len: int):
        """Return logical K,V [context_len,H,D], exclude trailing/padded slots."""
        raise NotImplementedError("M03: KVCache.gather")
