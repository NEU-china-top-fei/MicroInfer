"""Adapt per-call Context to attention kernels using only one layer's cache."""

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from microinfer.layer.metadata import AttentionMetadata
from microinfer.layer.paged_attention import paged_attention_meta
from microinfer.worker.kv_cache import LayerKVCache


class CachedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache: LayerKVCache | None = None

    def bind_cache(self, cache: LayerKVCache):
        self.cache = cache

    def forward(self, q, k, v, metadata=None):
        """Q/K already normalized/rotated. Return [T,Hq,D], including decode."""
        metadata = metadata or AttentionMetadata.from_context()
        cache = self.cache
        # Metadata construction + Runner debug validation own this invariant;
        # keeping the list comparison here would graph-break torch.compile.
        direct = metadata.direct_prefill if metadata.is_prefill else False
        if not direct and (cache is None or metadata.block_tables is None):
            raise RuntimeError(
                "paged attention requires a bound layer cache and block table"
            )
        if cache is not None:
            if metadata.slot_mapping is None:
                raise ValueError("bound cache requires slot_mapping")
            if metadata.validate_cache_write:
                cache.write(metadata.slot_mapping, k, v)
            else:
                blocks, offsets = metadata.slot_indices(cache.block_size)
                cache.write_fast(blocks, offsets, k, v)
        if not direct:
            return paged_attention_meta(q, cache.keys, cache.values, metadata)

        # No-prefix prefill/warmup: current K/V suffice. SDPA also works on CPU.
        # This eager path can later be replaced by flash_attn_varlen_func.
        result = torch.empty_like(q)
        if q.shape[1] % k.shape[1]:
            raise ValueError("query heads must be divisible by KV heads")
        repeats = q.shape[1] // k.shape[1]
        for start, end in pairwise(metadata.cu_q_cpu):
            if start == end:
                continue
            keys = k[start:end].repeat_interleave(repeats, dim=1).transpose(0, 1)
            values = v[start:end].repeat_interleave(repeats, dim=1).transpose(0, 1)
            result[start:end] = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1),
                keys,
                values,
                is_causal=True,
            ).transpose(0, 1)
        return result
