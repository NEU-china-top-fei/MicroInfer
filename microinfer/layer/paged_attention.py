import math

import torch

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


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Causal GQA over paged KV that already includes this step's new K/V.

    Q:
        [total_queries, Hq, D]

    K/V:
        [physical_blocks, B, Hkv, D]

    Return:
        same shape as Q.

    Query j in request r can attend through:

        context_len[r] - query_len[r] + j

    inclusive.

    Padded block IDs beyond context_len are ignored.

    Invalid metadata raises ValueError.
    """

    # ============================================================
    # Basic shapes
    # ============================================================

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

    (
        num_physical_blocks,
        block_size,
        num_kv_heads,
        kv_head_dim,
    ) = key_cache.shape

    num_requests = context_lens.numel()

    # ============================================================
    # Device / dtype relationships
    # ============================================================

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

    valid_index_dtypes = {
        torch.int32,
        torch.int64,
    }

    if block_tables.dtype not in valid_index_dtypes:
        raise ValueError("block_tables must be int32 or int64")

    if context_lens.dtype not in valid_index_dtypes:
        raise ValueError("context_lens must be int32 or int64")

    if cu_seqlens_q.dtype not in valid_index_dtypes:
        raise ValueError("cu_seqlens_q must be int32 or int64")

    # ============================================================
    # Head metadata
    # ============================================================

    if kv_head_dim != head_dim:
        raise ValueError("query and KV cache must have the same head dimension")

    if num_kv_heads <= 0:
        raise ValueError("number of KV heads must be positive")

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"Hq={num_q_heads} must be divisible by Hkv={num_kv_heads}")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # ============================================================
    # Request metadata
    # ============================================================

    if block_tables.shape[0] != num_requests:
        raise ValueError("block_tables.shape[0] must equal context_lens.numel()")

    if cu_seqlens_q.numel() != num_requests + 1:
        raise ValueError("cu_seqlens_q must contain num_requests + 1 entries")

    # Metadata is tiny compared with KV cache.
    # Converting to Python here keeps validation simple and explicit.
    cu_q = [int(x) for x in cu_seqlens_q.tolist()]

    contexts = [int(x) for x in context_lens.tolist()]

    if cu_q[0] != 0:
        raise ValueError("cu_seqlens_q[0] must be 0")

    if cu_q[-1] != total_queries:
        raise ValueError("cu_seqlens_q[-1] must equal query.shape[0]")

    q_lens: list[int] = []

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

        # Number of logical blocks that are actually valid.
        required_blocks = (context_len + block_size - 1) // block_size

        if required_blocks > block_tables.shape[1]:
            raise ValueError(
                f"request {i}: context_len={context_len} "
                "requires more blocks than block_tables provides"
            )

        # IMPORTANT:
        #
        # Only validate block IDs corresponding to actual context.
        # Trailing padded values may be -1 or arbitrary poison.
        if required_blocks:
            ids = block_tables[i, :required_blocks].tolist()

            for block_id in ids:
                block_id = int(block_id)

                if block_id < 0:
                    raise ValueError(
                        f"request {i}: invalid negative block id {block_id}"
                    )

                if block_id >= num_physical_blocks:
                    raise ValueError(
                        f"request {i}: block id {block_id} "
                        f"is outside [0, {num_physical_blocks})"
                    )

    # ============================================================
    # Empty input
    # ============================================================

    if total_queries == 0:
        return torch.empty_like(query)

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # ============================================================
    # Decide whether FlashAttention can actually be used
    # ============================================================

    # Small blocks/heads and CPU inputs use the reference implementation.
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
        # Avoid obscure zero-length edge cases in the CUDA path.
        and all(q_len > 0 for q_len in q_lens)
        and all(ctx > 0 for ctx in contexts)
    )

    if not can_use_flash:
        return _paged_attention_reference(
            query,
            key_cache,
            value_cache,
            block_tables,
            contexts,
            cu_q,
            scale=scale,
        )

    # ============================================================
    # FlashAttention metadata
    # ============================================================

    block_tables_i32 = block_tables.to(dtype=torch.int32).contiguous()

    context_lens_i32 = context_lens.to(dtype=torch.int32).contiguous()

    cu_seqlens_q_i32 = cu_seqlens_q.to(dtype=torch.int32).contiguous()

    cu_seqlens_k = torch.empty(
        num_requests + 1,
        dtype=torch.int32,
        device=query.device,
    )

    cu_seqlens_k[0] = 0

    cu_seqlens_k[1:] = torch.cumsum(
        context_lens_i32,
        dim=0,
        dtype=torch.int32,
    )

    # ============================================================
    # Decode fast path
    # ============================================================

    if all(q_len == 1 for q_len in q_lens):
        # Since every request contributes one query:
        #
        # [T, Hq, D] == [batch, Hq, D]
        #
        # ->
        #
        # [batch, 1, Hq, D]
        out = flash_attn_with_kvcache(
            query.unsqueeze(1),
            key_cache,
            value_cache,
            cache_seqlens=context_lens_i32,
            block_table=block_tables_i32,
            softmax_scale=scale,
            causal=True,
        )

        return out.squeeze(1)

    # ============================================================
    # Prefill / chunked-prefill
    # ============================================================

    max_seqlen_q = max(q_lens)
    max_seqlen_k = max(contexts)

    return flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_seqlens_q_i32,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables_i32,
    )
