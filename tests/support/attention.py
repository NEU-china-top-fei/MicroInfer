"""Test-only validated paged-attention entry point."""

from __future__ import annotations

import math

import torch

from microinfer.layer.metadata import AttentionMetadata
from microinfer.layer.paged_attention import paged_attention_meta


def checked_paged_attention(
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lens,
    cu_seqlens_q,
    *,
    scale=None,
):
    if query.ndim != 3:
        raise ValueError("query must have shape [T, Hq, D]")
    if key_cache.ndim != 4:
        raise ValueError("key_cache must have shape [num_blocks, block_size, Hkv, D]")
    if value_cache.ndim != 4:
        raise ValueError("value_cache must have shape [num_blocks, block_size, Hkv, D]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must have the same shape")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [num_requests, max_blocks]")
    if context_lens.ndim != 1:
        raise ValueError("context_lens must be 1-D")
    if cu_seqlens_q.ndim != 1:
        raise ValueError("cu_seqlens_q must be 1-D")

    total_queries, num_q_heads, head_dim = query.shape
    num_physical_blocks, block_size, num_kv_heads, kv_head_dim = key_cache.shape
    num_requests = context_lens.numel()

    if key_cache.device != query.device:
        raise ValueError("key_cache must be on the same device as query")
    if value_cache.device != query.device:
        raise ValueError("value_cache must be on the same device as query")
    if block_tables.device != query.device:
        raise ValueError("block_tables must be on the same device as query")
    if context_lens.device != query.device:
        raise ValueError("context_lens must be on the same device as query")
    if cu_seqlens_q.device != query.device:
        raise ValueError("cu_seqlens_q must be on the same device as query")
    if not query.dtype.is_floating_point:
        raise ValueError("query must have a floating-point dtype")
    if key_cache.dtype != query.dtype:
        raise ValueError("key_cache dtype must match query dtype")
    if value_cache.dtype != query.dtype:
        raise ValueError("value_cache dtype must match query dtype")
    valid_index_dtypes = {torch.int32, torch.int64}
    if block_tables.dtype not in valid_index_dtypes:
        raise ValueError("block_tables must be int32 or int64")
    if context_lens.dtype not in valid_index_dtypes:
        raise ValueError("context_lens must be int32 or int64")
    if cu_seqlens_q.dtype not in valid_index_dtypes:
        raise ValueError("cu_seqlens_q must be int32 or int64")
    if kv_head_dim != head_dim:
        raise ValueError("query and KV cache must have the same head dimension")
    if num_kv_heads <= 0:
        raise ValueError("number of KV heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"Hq={num_q_heads} must be divisible by Hkv={num_kv_heads}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_tables.shape[0] != num_requests:
        raise ValueError("block_tables.shape[0] must equal context_lens.numel()")
    if cu_seqlens_q.numel() != num_requests + 1:
        raise ValueError("cu_seqlens_q must contain num_requests + 1 entries")

    cu_q = [int(x) for x in cu_seqlens_q.tolist()]
    contexts = [int(x) for x in context_lens.tolist()]
    if cu_q[0] != 0:
        raise ValueError("cu_seqlens_q[0] must be 0")
    if cu_q[-1] != total_queries:
        raise ValueError("cu_seqlens_q[-1] must equal query.shape[0]")

    q_lens = []
    for i in range(num_requests):
        if cu_q[i + 1] < cu_q[i]:
            raise ValueError("cu_seqlens_q must be nondecreasing")
        query_len = cu_q[i + 1] - cu_q[i]
        context_len = contexts[i]
        if context_len < 0:
            raise ValueError("context_lens must be nonnegative")
        if query_len > context_len:
            raise ValueError(
                f"request {i}: query_len={query_len} exceeds context_len={context_len}"
            )
        q_lens.append(query_len)
        required_blocks = (context_len + block_size - 1) // block_size
        if required_blocks > block_tables.shape[1]:
            raise ValueError(
                f"request {i}: context_len={context_len} "
                "requires more blocks than block_tables provides"
            )
        for block_id in block_tables[i, :required_blocks].tolist():
            block_id = int(block_id)
            if block_id < 0:
                raise ValueError(f"request {i}: invalid negative block id {block_id}")
            if block_id >= num_physical_blocks:
                raise ValueError(
                    f"request {i}: block id {block_id} "
                    f"is outside [0, {num_physical_blocks})"
                )

    if total_queries == 0:
        return torch.empty_like(query)
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    is_decode = all(length == 1 for length in q_lens)
    cu_k = None
    if not is_decode:
        cumulative = [0]
        for length in contexts:
            cumulative.append(cumulative[-1] + length)
        cu_k = torch.tensor(cumulative, dtype=torch.int32, device=query.device)

    metadata = AttentionMetadata(
        is_prefill=not is_decode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(contexts),
        slot_mapping=None,
        context_lens=context_lens,
        block_tables=block_tables,
        direct_prefill=False,
        validate_cache_write=False,
        _cu_q_cpu=tuple(cu_q),
        _context_lens_cpu=tuple(contexts),
        _q_lens_cpu=tuple(q_lens),
    )
    return paged_attention_meta(query, key_cache, value_cache, metadata, scale=scale)
