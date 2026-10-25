import pytest
import torch

from microinfer.layer.paged_attention import _paged_attention_reference
from microinfer.layer.triton_paged_attention import HAS_TRITON, triton_paged_decode

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
