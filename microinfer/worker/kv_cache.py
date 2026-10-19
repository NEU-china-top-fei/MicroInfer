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
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.keys = torch.zeros(
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.values = torch.zeros(
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

    @property
    def num_bytes(self) -> int:
        number = (
            self.num_layers
            * self.num_blocks
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * 2
        )
        if self.dtype == torch.float32:
            factor = 4
        elif self.dtype == torch.bfloat16:
            factor = 2
        elif self.dtype == torch.float16:
            factor = 2

        return factor * number

    def write(
        self, layer: int, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Write [T,H,D] to flattened physical slots; -1 skips padding.

        Validate the entire call before mutation, including duplicate nonnegative
        slots, bounds and tensor shapes. Reject invalid calls with ValueError.
        """
        # negative one means skip
        invalid_nega = torch.any(slots < -1).item()
        if invalid_nega:
            raise ValueError("index less than -1")
        judge = slots >= 0
        slots = slots[judge]
        keys = keys[judge, :, :]
        values = values[judge, :, :]
        # duplicate or negative
        is_dup = len(slots.unique()) < slots.shape[0]
        if is_dup:
            raise ValueError("duplicate slot value")

        # bound
        if slots.max().item() >= self.num_blocks * self.block_size:
            raise ValueError("bound overflowed")
        # tensor shape
        kposition, knum_heads, khead_dim = keys.shape
        vposition, vnum_heads, vhead_dim = values.shape
        equal = (
            knum_heads == vnum_heads == self.num_kv_heads
            and khead_dim == vhead_dim == self.head_dim
        )
        if not equal:
            raise ValueError("reshape mismatch")
        # breakpoint()

        self.keys[layer, slots // self.block_size, slots % self.block_size, :, :] = (
            keys.to(self.keys.dtype)
        )
        self.values[layer, slots // self.block_size, slots % self.block_size, :, :] = (
            values.to(self.values.dtype)
        )
        # different type:origin to cache(float32->bf16)

    def gather(self, layer: int, block_table: list[int], context_len: int):
        """Return logical K,V [context_len,H,D], exclude trailing/padded slots."""

        """
        e.g:
        [3,2,5]T@[4,4,4,4]+[0,1,2,3] get the actual idx matrix,flatten can get the true
        """
        # breakpoint()
        if context_len < 0:
            raise ValueError
        len_table = len(block_table)
        if len_table == 0:
            raise ValueError("no valid table")
        block_idx = torch.tensor(block_table, dtype=torch.int64).reshape((len_table, 1))
        strides = torch.tensor([self.block_size] * self.block_size).reshape(
            (1, self.block_size)
        )
        idx_matrix = torch.matmul(block_idx, strides) + torch.arange(0, self.block_size)
        if context_len > len_table * self.block_size:
            raise ValueError("index overflow")
        true_idx = idx_matrix.flatten()[:context_len]
        judge = torch.any(true_idx >= self.block_size * self.num_blocks).item()
        if judge:
            raise ValueError("index overflow")

        keys_cached = self.keys[
            layer, true_idx // self.block_size, true_idx % self.block_size, :, :
        ]
        value_cached = self.values[
            layer, true_idx // self.block_size, true_idx % self.block_size, :, :
        ]

        return keys_cached, value_cached
