"""Attention over one layer's KV cache, driven by AttentionMetadata."""

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from microinfer.layer.paged_attention import (
    _index_tensor,
    flash_attn_varlen_func,
    paged_attention_meta,
)
from microinfer.worker.kv_cache import LayerKVCache


def _direct_prefill_flash(q, k, v, metadata):
    """FlashAttention varlen for prefix-free prefill, when supported."""
    head_dim = q.shape[-1]
    if (
        flash_attn_varlen_func is None
        or not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16)
        or head_dim > 256
        or head_dim % 8
        or metadata.cu_seqlens_q is None
        or metadata.cu_seqlens_k is None
        or metadata.max_seqlen_q <= 0
        or metadata.max_seqlen_k <= 0
    ):
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=_index_tensor(metadata.cu_seqlens_q),
        cu_seqlens_k=_index_tensor(metadata.cu_seqlens_k),
        max_seqlen_q=metadata.max_seqlen_q,
        max_seqlen_k=metadata.max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )


def _direct_prefill_sdpa(q, k, v):
    """GQA SDPA with a contiguous [1, H, L, D] layout (no repeat_interleave)."""
    query = q.transpose(0, 1).contiguous().unsqueeze(0)
    keys = k.transpose(0, 1).contiguous().unsqueeze(0)
    values = v.transpose(0, 1).contiguous().unsqueeze(0)
    out = F.scaled_dot_product_attention(
        query, keys, values, is_causal=True, enable_gqa=True
    )
    return out.squeeze(0).transpose(0, 1)


class CachedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache: LayerKVCache | None = None

    def bind_cache(self, cache: LayerKVCache):
        self.cache = cache

    def forward(self, q, k, v, metadata):
        """Q/K already normalized/rotated. Return [T,Hq,D], including decode."""
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
                cache.write_checked(metadata.slot_mapping, k, v)
            else:
                blocks, offsets = metadata.slot_indices(cache.block_size)
                cache.write_fast(blocks, offsets, k, v)
        if not direct:
            return paged_attention_meta(q, cache.keys, cache.values, metadata)

        # No-prefix prefill/warmup: current K/V suffice. FlashAttention varlen
        # handles packed requests in one call; SDPA is the CPU/FP32 fallback.
        if q.shape[1] % k.shape[1]:
            raise ValueError("query heads must be divisible by KV heads")
        flashed = _direct_prefill_flash(q, k, v, metadata)
        if flashed is not None:
            return flashed
        result = torch.empty_like(q)
        for start, end in pairwise(metadata.cu_q_cpu):
            if start == end:
                continue
            result[start:end] = _direct_prefill_sdpa(
                q[start:end], k[start:end], v[start:end]
            )
        return result
