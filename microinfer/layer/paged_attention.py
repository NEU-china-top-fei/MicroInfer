import math

import torch

from microinfer.layer.metadata import AttentionMetadata
from microinfer.layer.triton_paged_attention import (
    HAS_TRITON,
    triton_paged_decode,
    triton_paged_prefill,
)

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


def _gather_paged_kv(key_cache, value_cache, block_tables, contexts, block_size):
    """Materialize the valid rows of each paged request into varlen K/V.

    FlashAttention's paged varlen kernel requires block_size % 256 == 0. For
    smaller blocks we copy the referenced rows once per layer and call the
    regular varlen kernel. The copy is prefill-only and tiny relative to the
    GEMMs, while one kernel call replaces one Python loop per request.
    """
    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    keys = []
    values = []
    for request_idx, context_len in enumerate(contexts):
        num_blocks = (context_len + block_size - 1) // block_size
        blocks = block_tables[request_idx, :num_blocks]
        keys.append(
            key_cache.index_select(0, blocks).reshape(-1, num_kv_heads, head_dim)[
                :context_len
            ]
        )
        values.append(
            value_cache.index_select(0, blocks).reshape(-1, num_kv_heads, head_dim)[
                :context_len
            ]
        )
    return torch.cat(keys), torch.cat(values)


# Reading the cache in place wins once the copied prefix dwarfs the queries;
# for a full-length prompt FlashAttention's gather copy stays faster.
TRITON_PREFILL_RATIO = 4


def _paged_prefill_triton(query, key_cache, value_cache, metadata, scale):
    """Prefix-heavy prefill without materializing the paged K/V."""
    if (
        not HAS_TRITON
        or not query.is_cuda
        or query.dtype not in (torch.float16, torch.bfloat16)
        or query.shape[-1] % 16
        or metadata.block_tables is None
        or metadata.cu_seqlens_q is None
        or metadata.cu_seqlens_k is None
        or metadata.max_seqlen_q <= 0
    ):
        return None
    contexts = metadata.context_lens_cpu
    q_lens = metadata.q_lens_cpu
    if not contexts or len(contexts) != len(q_lens):
        return None
    if any(length <= 0 for length in contexts):
        return None
    if sum(contexts) < TRITON_PREFILL_RATIO * sum(q_lens):
        return None
    return triton_paged_prefill(
        query,
        key_cache,
        value_cache,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.block_tables,
        max_seqlen_q=metadata.max_seqlen_q,
        scale=scale,
    )


def _paged_prefill_gather(query, key_cache, value_cache, metadata, scale):
    """Varlen prefill for paged caches whose block size FlashAttention rejects."""
    if (
        flash_attn_varlen_func is None
        or not query.is_cuda
        or query.dtype not in (torch.float16, torch.bfloat16)
        or query.shape[-1] > 256
        or query.shape[-1] % 8
        or metadata.block_tables is None
        or metadata.cu_seqlens_q is None
        or metadata.cu_seqlens_k is None
    ):
        return None
    contexts = metadata.context_lens_cpu
    q_lens = metadata.q_lens_cpu
    if (
        not contexts
        or not q_lens
        or len(contexts) != len(q_lens)
        or any(length <= 0 for length in contexts)
        or any(length <= 0 for length in q_lens)
        or any(q_len > context_len for q_len, context_len in zip(q_lens, contexts))
    ):
        return None

    keys, values = _gather_paged_kv(
        key_cache,
        value_cache,
        metadata.block_tables,
        contexts,
        key_cache.shape[1],
    )
    return flash_attn_varlen_func(
        query,
        keys,
        values,
        cu_seqlens_q=_index_tensor(metadata.cu_seqlens_q),
        cu_seqlens_k=_index_tensor(metadata.cu_seqlens_k),
        max_seqlen_q=metadata.max_seqlen_q,
        max_seqlen_k=metadata.max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
    )


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

    if metadata.is_decode:
        context_lens = metadata.context_lens
        block_tables = metadata.block_tables
        if context_lens is not None and block_tables is not None:
            if can_use_flash:
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
            if (
                HAS_TRITON
                and query.is_cuda
                and query.shape[1] % key_cache.shape[2] == 0
                and block_size & (block_size - 1) == 0
                and head_dim & (head_dim - 1) == 0
            ):
                return triton_paged_decode(
                    query,
                    key_cache,
                    value_cache,
                    block_tables,
                    context_lens,
                    scale=scale,
                )

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

    if metadata.is_prefill:
        paged = _paged_prefill_triton(query, key_cache, value_cache, metadata, scale)
        if paged is not None:
            return paged
        gathered = _paged_prefill_gather(query, key_cache, value_cache, metadata, scale)
        if gathered is not None:
            return gathered

    return _paged_attention_reference(
        query,
        key_cache,
        value_cache,
        metadata.block_tables,
        list(metadata.context_lens_cpu),
        list(metadata.cu_q_cpu),
        scale=scale,
    )
