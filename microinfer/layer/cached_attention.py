"""Adapt per-call Context to attention kernels using only one layer's cache."""

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from microinfer.layer.paged_attention import paged_attention
from microinfer.utils.context import get_context
from microinfer.worker.kv_cache import LayerKVCache


class CachedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache: LayerKVCache | None = None

    def bind_cache(self, cache: LayerKVCache):
        self.cache = cache

    def forward(self, q, k, v):
        """Q/K already normalized/rotated. Return [T,Hq,D], including decode."""
        ctx = get_context()
        cache = self.cache
        if ctx.is_prefill:
            cu_q = ctx.cu_seqlens_q
            lengths = ctx.cu_seqlens_k[1:] - ctx.cu_seqlens_k[:-1]
            direct = ctx.block_tables is None
            if direct and not torch.equal(lengths, cu_q[1:] - cu_q[:-1]):
                raise ValueError("cached prefixes require block_tables")
        else:
            lengths = ctx.context_lens
            cu_q = torch.arange(q.shape[0] + 1, dtype=torch.int32, device=q.device)
            direct = False
        if not direct and (cache is None or ctx.block_tables is None):
            raise RuntimeError(
                "paged attention requires a bound layer cache and block table"
            )
        if cache is not None:
            if ctx.slot_mapping is None:
                raise ValueError("bound cache requires slot_mapping")
            cache.write(ctx.slot_mapping, k, v)
        if not direct:
            return paged_attention(
                q, cache.keys, cache.values, ctx.block_tables, lengths, cu_q
            )

        # No-prefix prefill/warmup: current K/V suffice. SDPA also works on CPU.
        # This eager path can later be replaced by flash_attn_varlen_func.
        result = torch.empty_like(q)
        offsets = cu_q.tolist()
        if q.shape[1] % k.shape[1]:
            raise ValueError("query heads must be divisible by KV heads")
        repeats = q.shape[1] // k.shape[1]
        for start, end in pairwise(offsets):
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
