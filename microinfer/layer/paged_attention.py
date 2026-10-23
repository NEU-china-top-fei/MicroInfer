import math

import torch

from microinfer.layer.metadata import AttentionMetadata

try:
    from flash_attn import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


def _paged_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: list[int],
    cu_seqlens_q: list[int],
    *,
    scale: float,
) -> torch.Tensor:
    _total_query, num_query_head, head_dim = query.shape
    _num_blocks, block_size, num_kv_head, _ = key_cache.shape
    num_request = block_tables.shape[0]
    result = torch.zeros_like(query)
    # Half-precision QK can overflow before softmax. Keep both contractions and
    # masking in one accumulation dtype, then restore the activation dtype.
    accumulation_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    for request_idx in range(num_request):
        context_len = context_lens[request_idx]
        block_num = (context_len + block_size - 1) // block_size
        blocks = block_tables[request_idx, :block_num]  # physical block id
        keys = key_cache[blocks].reshape(-1, num_kv_head, head_dim)[:context_len]
        values = value_cache[blocks].reshape(-1, num_kv_head, head_dim)[:context_len]
        keys = keys.to(accumulation_dtype)
        values = values.to(accumulation_dtype)
        start = cu_seqlens_q[request_idx]
        end = cu_seqlens_q[request_idx + 1]
        cur_query = (
            query[start:end]
            .to(accumulation_dtype)
            .reshape(-1, num_kv_head, num_query_head // num_kv_head, head_dim)
        )
        score = torch.einsum("qhgd,khd->hgqk", cur_query, keys)
        prefix_len = context_len - (end - start)
        qposition = prefix_len + torch.arange(end - start, device=query.device)
        # start means the start idx in query vector,there we need position idx in seq
        kposition = torch.arange(context_len, device=query.device)
        truth_table = qposition.unsqueeze(1) >= kposition.unsqueeze(0)
        scores = (score * scale).masked_fill(~truth_table, float("-inf"))
        out = torch.einsum("hgqk,khd->qhgd", torch.softmax(scores, dim=-1), values)

        result[start:end] = out.reshape(-1, num_query_head, head_dim).to(query.dtype)

    return result


def _index_tensor(tensor: torch.Tensor, dtype=torch.int32) -> torch.Tensor:
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor.contiguous()


def paged_attention_meta(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: AttentionMetadata,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Kernel dispatch for validated metadata; no host validation or tolist."""
    if query.shape[0] == 0:
        return torch.empty_like(query)
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    _, _, head_dim = query.shape
    block_size = key_cache.shape[1]
    can_use_flash = (
        flash_attn_varlen_func is not None
        and flash_attn_with_kvcache is not None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and head_dim <= 256
        and head_dim % 8 == 0
        and block_size % 256 == 0
        and query.stride(-1) == 1
        and key_cache.stride(-1) == 1
        and value_cache.stride(-1) == 1
    )

    if can_use_flash and metadata.is_decode:
        context_lens = metadata.context_lens
        block_tables = metadata.block_tables
        if context_lens is not None and block_tables is not None:
            out = flash_attn_with_kvcache(
                query.unsqueeze(1),
                key_cache,
                value_cache,
                cache_seqlens=_index_tensor(context_lens),
                block_table=_index_tensor(block_tables),
                softmax_scale=scale,
                causal=True,
            )
            return out.squeeze(1)

    if can_use_flash and metadata.is_prefill:
        cu_q = metadata.cu_seqlens_q
        cu_k = metadata.cu_seqlens_k
        block_tables = metadata.block_tables
        q_lens = metadata.q_lens_cpu
        contexts = metadata.context_lens_cpu
        if (
            cu_q is not None
            and cu_k is not None
            and block_tables is not None
            and q_lens
            and contexts
            and all(length > 0 for length in q_lens)
            and all(length > 0 for length in contexts)
        ):
            return flash_attn_varlen_func(
                query,
                key_cache,
                value_cache,
                cu_seqlens_q=_index_tensor(cu_q),
                cu_seqlens_k=_index_tensor(cu_k),
                max_seqlen_q=metadata.max_seqlen_q,
                max_seqlen_k=metadata.max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=True,
                block_table=_index_tensor(block_tables),
            )

    return _paged_attention_reference(
        query,
        key_cache,
        value_cache,
        metadata.block_tables,
        list(metadata.context_lens_cpu),
        list(metadata.cu_q_cpu),
        scale=scale,
    )
