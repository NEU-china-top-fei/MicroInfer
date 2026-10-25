"""Triton paged decode attention for arbitrary block sizes.

FlashAttention 2's paged decode kernel requires ``block_size % 256 == 0``.
This kernel keeps the same paged KV layout but works for 16/32/64/128/256
blocks and for CUDA float32 as well, so it can replace the slow Python
reference path (and be captured by CUDA Graph).

Each program walks a tile of logical positions (128 under eager/CUDA Graph,
32 under torch.compile to avoid Inductor register pressure) and gathers block
ids for the whole tile, instead of looping once per physical block.
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
        out_ptr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        BLOCK_N: tl.constexpr,
        block_table_width,
        scale,
    ):
        request = tl.program_id(0)
        head = tl.program_id(1)
        group_size = num_q_heads // num_kv_heads
        kv_head = head // group_size

        context_len = tl.load(context_lens_ptr + request)
        offsets_d = tl.arange(0, head_dim)
        offsets_n = tl.arange(0, BLOCK_N)
        q = tl.load(
            q_ptr + request * num_q_heads * head_dim + head * head_dim + offsets_d
        ).to(tl.float32)

        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((head_dim,), dtype=tl.float32)

        # Process BLOCK_N logical positions per iteration. A tile may span
        # several physical blocks, so gather up to BLOCK_N block ids at once.
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
            ).to(tl.float32)
            scores = tl.sum(key * q[None, :], axis=1) * scale
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

        result = accumulator / running_sum
        tl.store(
            out_ptr + request * num_q_heads * head_dim + head * head_dim + offsets_d,
            result,
        )


def triton_paged_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode one query per request over paged KV; arbitrary block size."""
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

    # A wide tile reduces loop iterations and wins for eager/CUDA Graph. Under
    # Inductor the same tile spills registers, so use a narrower one.
    block_n = 32 if torch.compiler.is_compiling() else 128
    grid = (num_requests, num_q_heads)
    _paged_decode_kernel[grid](
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        output,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        block_n,
        block_tables.shape[1],
        scale,
        num_warps=4,
        num_stages=2,
    )
    return output
