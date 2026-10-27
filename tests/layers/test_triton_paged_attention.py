import pytest
import torch

from microinfer.layer.paged_attention import _paged_attention_reference
from microinfer.layer.triton_paged_attention import (
    HAS_TRITON,
    triton_paged_decode,
    triton_paged_prefill,
)

pytestmark = [pytest.mark.cuda, pytest.mark.integration]


@pytest.mark.parametrize("block_size", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_paged_decode_matches_reference(block_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    if not HAS_TRITON:
        pytest.skip("Triton is not available")

    torch.manual_seed(0)
    num_requests, num_q_heads, num_kv_heads, head_dim = 2, 4, 2, 32
    contexts = [5, block_size + 3]
    num_blocks = (max(contexts) + block_size - 1) // block_size

    physical = list(range(num_requests * num_blocks))[::-1]
    block_tables = torch.tensor(
        [physical[:num_blocks], physical[num_blocks : 2 * num_blocks]],
        dtype=torch.int32,
        device="cuda",
    )
    key_cache = torch.randn(
        num_requests * num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(num_requests, num_q_heads, head_dim, device="cuda", dtype=dtype)

    for request, context_len in enumerate(contexts):
        for position in range(context_len):
            block_id = int(block_tables[request, position // block_size])
            key_cache[block_id, position % block_size] = torch.randn(
                num_kv_heads, head_dim, device="cuda", dtype=dtype
            )
            value_cache[block_id, position % block_size] = torch.randn(
                num_kv_heads, head_dim, device="cuda", dtype=dtype
            )

    scale = 1.0 / head_dim**0.5
    expected = _paged_attention_reference(
        query,
        key_cache,
        value_cache,
        block_tables,
        contexts,
        [0, 1, 2],
        scale=scale,
    )
    actual = triton_paged_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        torch.tensor(contexts, dtype=torch.int32, device="cuda"),
        scale=scale,
    )

    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=tolerance, rtol=tolerance
    )
    assert actual.dtype == dtype


@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_paged_prefill_matches_reference(block_size, dtype):
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA Triton is not available")

    torch.manual_seed(0)
    num_q_heads, num_kv_heads, head_dim = 4, 2, 32
    contexts = [block_size + 5, 2 * block_size + 9]
    q_lens = [19, 7]
    requests = len(contexts)
    max_blocks = max((c + block_size - 1) // block_size for c in contexts)

    block_tables = torch.arange(requests * max_blocks, dtype=torch.int32, device="cuda")
    block_tables = block_tables.reshape(requests, max_blocks).flip(-1).contiguous()
    key_cache = torch.zeros(
        requests * max_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    value_cache = torch.zeros_like(key_cache)
    for request, context in enumerate(contexts):
        for position in range(context):
            block = int(block_tables[request, position // block_size])
            key_cache[block, position % block_size] = torch.randn(
                num_kv_heads, head_dim, device="cuda", dtype=dtype
            )
            value_cache[block, position % block_size] = torch.randn(
                num_kv_heads, head_dim, device="cuda", dtype=dtype
            )

    total_q = sum(q_lens)
    query = torch.randn(total_q, num_q_heads, head_dim, device="cuda", dtype=dtype)
    cu_q, cu_k = [0], [0]
    for q_len, context in zip(q_lens, contexts):
        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + context)

    scale = 1.0 / head_dim**0.5
    expected = _paged_attention_reference(
        query,
        key_cache,
        value_cache,
        block_tables,
        contexts,
        cu_q,
        scale=scale,
    )
    actual = triton_paged_prefill(
        query,
        key_cache,
        value_cache,
        torch.tensor(cu_q, dtype=torch.int32, device="cuda"),
        torch.tensor(cu_k, dtype=torch.int32, device="cuda"),
        block_tables,
        max_seqlen_q=max(q_lens),
        scale=scale,
    )

    torch.testing.assert_close(
        actual.float(), expected.float(), atol=2e-2, rtol=2e-2
    )
    assert actual.dtype == dtype
