"""Triton KV-cache scatter used by the decode and prefill hot paths."""

from __future__ import annotations

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
    def _store_kv_kernel(
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        blocks_ptr,
        offsets_ptr,
        key_stride,
        value_stride,
        block_size,
        HD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.load(blocks_ptr + token)
        offset = tl.load(offsets_ptr + token)
        lanes = tl.arange(0, BLOCK)
        mask = lanes < HD
        source = token * key_stride + lanes
        key = tl.load(key_ptr + source, mask=mask)
        value = tl.load(value_ptr + token * value_stride + lanes, mask=mask)
        destination = (block * block_size + offset) * HD + lanes
        tl.store(key_cache_ptr + destination, key, mask=mask)
        tl.store(value_cache_ptr + destination, value, mask=mask)


def store_kv(
    key_cache,
    value_cache,
    blocks,
    offsets,
    keys,
    values,
    block_size: int,
):
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    tokens, heads, head_dim = keys.shape
    hd = heads * head_dim
    if values.shape != keys.shape:
        raise ValueError("K/V shapes differ")
    if keys.stride(-1) != 1 or keys.stride(-2) != head_dim:
        raise ValueError("K/V inner dimensions must be contiguous")
    if values.stride(-1) != 1 or values.stride(-2) != head_dim:
        raise ValueError("V inner dimensions must be contiguous")
    width = triton.next_power_of_2(hd)
    _store_kv_kernel[(tokens,)](
        keys,
        values,
        key_cache,
        value_cache,
        blocks,
        offsets,
        keys.stride(0),
        values.stride(0),
        block_size,
        hd,
        width,
        num_warps=4,
    )
