"""Triton paged decode attention for arbitrary block sizes.

FlashAttention 2's paged decode kernel requires ``block_size % 256 == 0``.
This kernel keeps the same paged KV layout but works for 16/32/64/128/256
blocks and for CUDA float32 as well, so it can replace the slow Python
reference path (and be captured by CUDA Graph).

Each program owns one ``(request, query head)`` pair and walks a tile of
logical positions (128 under eager/CUDA Graph, 32 under torch.compile to avoid
Inductor register pressure), gathering block ids for the whole tile instead of
looping once per physical block. When the batch cannot fill the GPU, the scan
is striped over ``SPLIT_K`` programs and a small second kernel combines the
per-split online-softmax states.

``triton_paged_prefill`` applies the same paged addressing to prefill: a
BLOCK_M query tile attends to every cached key of its request, so prefix-heavy
prefill needs no gather copy.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _paged_decode_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        block_table_ptr,
        context_lens_ptr,
        partial_acc_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
        DIRECT: tl.constexpr,
        block_table_width,
        scale,
    ):
        request = tl.program_id(0)
        head = tl.program_id(1)
        split = tl.program_id(2)
        group_size = num_q_heads // num_kv_heads
        kv_head = head // group_size

        context_len = tl.load(context_lens_ptr + request)
        offsets_d = tl.arange(0, head_dim)
        offsets_n = tl.arange(0, BLOCK_N)
        query = tl.load(
            q_ptr + request * num_q_heads * head_dim + head * head_dim + offsets_d
        ).to(tl.float32)

        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((head_dim,), dtype=tl.float32)

        # Strided tile ownership keeps every split busy for any context length,
        # without a host-side read of the dynamic context_lens buffer.
        num_tiles = tl.cdiv(context_len, BLOCK_N)
        for tile in tl.range(split, num_tiles, SPLIT_K):
            positions = tile * BLOCK_N + offsets_n
            mask = positions < context_len
            block_ids = tl.load(
                block_table_ptr + request * block_table_width + positions // block_size,
                mask=mask,
                other=0,
            )
            in_block = positions - (positions // block_size) * block_size
            kv_offsets = (
                block_ids[:, None] * (block_size * num_kv_heads * head_dim)
                + in_block[:, None] * (num_kv_heads * head_dim)
                + kv_head * head_dim
                + offsets_d[None, :]
            )
            key = tl.load(
                k_ptr + kv_offsets,
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key * query[None, :], axis=1) * scale
            scores = tl.where(mask, scores, -float("inf"))

            new_max = tl.maximum(running_max, tl.max(scores))
            alpha = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max)
            running_sum = running_sum * alpha + tl.sum(probabilities)
            value = tl.load(
                v_ptr + kv_offsets,
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * alpha + tl.sum(
                probabilities[:, None] * value, axis=0
            )
            running_max = new_max

        if DIRECT:
            # One split: skip the workspace and the reduction launch entirely.
            tl.store(
                partial_acc_ptr
                + (request * num_q_heads + head) * head_dim
                + offsets_d,
                accumulator / running_sum,
            )
        else:
            state = (request * num_q_heads + head) * SPLIT_K + split
            tl.store(
                partial_acc_ptr + state * head_dim + offsets_d,
                accumulator,
            )
            tl.store(partial_max_ptr + state, running_max)
            tl.store(partial_sum_ptr + state, running_sum)

    @triton.jit
    def _paged_prefill_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        cu_q_ptr,
        cu_k_ptr,
        block_table_ptr,
        out_ptr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        block_table_width,
        scale,
    ):
        block_m = tl.program_id(0)
        head = tl.program_id(1)
        request = tl.program_id(2)

        q_start = tl.load(cu_q_ptr + request)
        q_end = tl.load(cu_q_ptr + request + 1)
        q_len = q_end - q_start
        rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        if block_m * BLOCK_M >= q_len:
            return
        row_valid = rows < q_len
        context_len = tl.load(cu_k_ptr + request + 1) - tl.load(cu_k_ptr + request)
        prefix = context_len - q_len
        q_positions = prefix + rows

        offsets_d = tl.arange(0, head_dim)
        offsets_n = tl.arange(0, BLOCK_N)
        group_size = num_q_heads // num_kv_heads
        kv_head = head // group_size
        query = tl.load(
            q_ptr
            + (q_start + rows[:, None]) * num_q_heads * head_dim
            + head * head_dim
            + offsets_d[None, :],
            mask=row_valid[:, None],
            other=0.0,
        )

        running_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

        for start_n in tl.range(0, context_len, BLOCK_N):
            positions = start_n + offsets_n
            mask = positions < context_len
            block_ids = tl.load(
                block_table_ptr + request * block_table_width + positions // block_size,
                mask=mask,
                other=0,
            )
            in_block = positions - (positions // block_size) * block_size
            kv_offsets = (
                block_ids[:, None] * (block_size * num_kv_heads * head_dim)
                + in_block[:, None] * (num_kv_heads * head_dim)
                + kv_head * head_dim
                + offsets_d[None, :]
            )
            key = tl.load(
                k_ptr + kv_offsets,
                mask=mask[:, None],
                other=0.0,
            )
            scores = tl.dot(query, tl.trans(key)) * scale
            causal = positions[None, :] <= q_positions[:, None]
            keep = causal & mask[None, :] & row_valid[:, None]
            scores = tl.where(keep, scores, -float("inf"))
            # Fully masked rows would make exp(-inf - -inf) NaN; they are
            # padded queries and their stores are masked below.
            scores = tl.where(row_valid[:, None], scores, 0.0)

            new_max = tl.maximum(running_max, tl.max(scores, axis=1))
            alpha = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
            value = tl.load(
                v_ptr + kv_offsets,
                mask=mask[:, None],
                other=0.0,
            )
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probabilities.to(value.dtype), value
            )
            running_max = new_max

        result = accumulator / running_sum[:, None]
        tl.store(
            out_ptr
            + (q_start + rows[:, None]) * num_q_heads * head_dim
            + head * head_dim
            + offsets_d[None, :],
            result,
            mask=row_valid[:, None],
        )

    @triton.jit
    def _paged_decode_reduce_kernel(
        partial_acc_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        out_ptr,
        head_dim: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        state = tl.program_id(0) * SPLIT_K + tl.arange(0, SPLIT_K)
        offsets_d = tl.arange(0, head_dim)
        partial_max = tl.load(partial_max_ptr + state)
        partial_sum = tl.load(partial_sum_ptr + state)
        # Empty splits report -inf/0 and contribute zero weight.
        global_max = tl.max(partial_max)
        weight = tl.exp(partial_max - global_max)
        denominator = tl.sum(partial_sum * weight)
        partial_acc = tl.load(
            partial_acc_ptr + state[:, None] * head_dim + offsets_d[None, :]
        )
        numerator = tl.sum(partial_acc * weight[:, None], axis=0)
        result = numerator / denominator
        tl.store(out_ptr + tl.program_id(0) * head_dim + offsets_d, result)


def triton_paged_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    max_seqlen_q: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal prefill straight from the paged cache (no gather copy).

    One program computes a BLOCK_M x head query tile against every cached key
    of its request, so the per-layer ``index_select``/``cat`` materialization
    is gone. This is used when the block size rules out FlashAttention's paged
    varlen kernel.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if query.ndim != 3:
        raise ValueError("query must have shape [tokens, Hq, D]")
    num_tokens, num_q_heads, head_dim = query.shape
    _num_blocks, block_size, num_kv_heads, kv_dim = key_cache.shape
    num_requests = block_tables.shape[0]
    if num_tokens == 0 or num_requests == 0:
        return torch.empty_like(query)
    if query.device != key_cache.device or query.dtype != key_cache.dtype:
        raise ValueError("query and cache must share device/dtype")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton paged prefill requires FP16/BF16")
    if kv_dim != head_dim or head_dim % 16:
        raise ValueError("head dimension must match and be a multiple of 16")
    if num_q_heads % num_kv_heads:
        raise ValueError("Hq must be divisible by Hkv")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Triton kernel requires a power-of-two block size")
    if cu_seqlens_q.numel() != num_requests + 1:
        raise ValueError("cu_seqlens_q must have num_requests + 1 entries")
    if cu_seqlens_k.numel() != num_requests + 1:
        raise ValueError("cu_seqlens_k must have num_requests + 1 entries")
    if max_seqlen_q <= 0:
        raise ValueError("max_seqlen_q must be positive")

    query = query.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(torch.int32).contiguous()
    block_tables = block_tables.to(torch.int32).contiguous()
    output = torch.empty_like(query)
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    block_m, block_n = 16, 32
    grid = (
        triton.cdiv(max_seqlen_q, block_m),
        num_q_heads,
        num_requests,
    )
    _paged_prefill_kernel[grid](
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        output,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        block_m,
        block_n,
        block_tables.shape[1],
        scale,
        num_warps=4,
        num_stages=2,
    )
    return output


DEFAULT_SPLIT_K = 4
# Below this many (request, head) programs the decode grid cannot fill the GPU,
# so long contexts split each head's scan across several programs.
MIN_PROGRAMS = 64


def _select_split_k(num_requests: int, num_q_heads: int) -> int:
    return DEFAULT_SPLIT_K if num_requests * num_q_heads < MIN_PROGRAMS else 1


def triton_paged_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: float | None = None,
    split_k: int | None = None,
) -> torch.Tensor:
    """Decode one query per request over paged KV; arbitrary block size.

    Each program owns one (request, query head) pair. When the batch is too
    small to fill the GPU, ``split_k`` programs stripe the context between them
    and a second kernel combines the online-softmax states. The split count is
    a launch-shape constant, so CUDA Graph can capture it even though context
    lengths change on replay.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if query.ndim != 3:
        raise ValueError("query must have shape [requests, Hq, D]")
    num_requests, num_q_heads, head_dim = query.shape
    _num_blocks, block_size, num_kv_heads, kv_dim = key_cache.shape
    if num_requests == 0:
        return torch.empty_like(query)
    if query.device != key_cache.device or query.dtype != key_cache.dtype:
        raise ValueError("query and cache must share device/dtype")
    if not query.is_cuda:
        raise ValueError("Triton paged decode requires CUDA")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported query dtype")
    if kv_dim != head_dim:
        raise ValueError("query and cache head dimensions differ")
    if num_q_heads % num_kv_heads:
        raise ValueError("Hq must be divisible by Hkv")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Triton kernel requires a power-of-two block size")
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("Triton kernel requires a power-of-two head_dim")
    if split_k is None:
        split_k = _select_split_k(num_requests, num_q_heads)
    if type(split_k) is not int or split_k <= 0 or split_k & (split_k - 1):
        raise ValueError("split_k must be a positive power of two")
    if block_tables.shape[0] != num_requests:
        raise ValueError("block_tables must have one row per request")
    if context_lens.numel() != num_requests:
        raise ValueError("context_lens must have one entry per request")

    query = query.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    block_tables = block_tables.to(torch.int32).contiguous()
    context_lens = context_lens.to(torch.int32).contiguous()
    output = torch.empty_like(query)
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if split_k == 1:
        # DIRECT=True writes the result to `workspace`; the reduction buffers
        # are unused, so reuse `output` instead of allocating them.
        workspace = output
        partial_max = partial_sum = output
    else:
        states = num_requests * num_q_heads * split_k
        workspace = torch.empty(
            (states, head_dim), dtype=torch.float32, device=query.device
        )
        partial_max = torch.empty(states, dtype=torch.float32, device=query.device)
        partial_sum = torch.empty(states, dtype=torch.float32, device=query.device)

    # A wide tile reduces loop iterations and wins for eager/CUDA Graph. Under
    # Inductor the same tile spills registers, so use a narrower one.
    block_n = 32 if torch.compiler.is_compiling() else 128
    _paged_decode_kernel[(num_requests, num_q_heads, split_k)](
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        workspace,
        partial_max,
        partial_sum,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        block_n,
        split_k,
        split_k == 1,
        block_tables.shape[1],
        scale,
        num_warps=4,
        num_stages=2,
    )
    if split_k > 1:
        _paged_decode_reduce_kernel[(num_requests * num_q_heads,)](
            workspace,
            partial_max,
            partial_sum,
            output,
            head_dim,
            split_k,
            num_warps=4,
        )
    return output
