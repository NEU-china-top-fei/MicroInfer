import pytest
import torch

from microinfer.layer.paged_attention import _paged_attention_reference
from tests.support.attention import checked_paged_attention as paged_attention

pytestmark = pytest.mark.component


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prefix,query_len", [(0, 3), (2, 1), (1, 2)])
def test_paged_causal_gqa_matches_dense_reference(
    prefix, query_len, seeded_torch, device, dtype
):
    # Physical order 2 -> 0 differs from logical order; trailing slots are poison.
    context = prefix + query_len
    q = torch.randn(query_len, 4, 4).to(device=device, dtype=dtype)
    logical_k = torch.randn(context, 2, 4).to(device=device, dtype=dtype)
    logical_v = torch.randn(context, 2, 4).to(device=device, dtype=dtype)
    k = torch.full((3, 2, 2, 4), 1000.0, device=device, dtype=dtype)
    v = torch.full_like(k, -1000)
    table = [2, 0]
    for pos in range(context):
        k[table[pos // 2], pos % 2] = logical_k[pos]
        v[table[pos // 2], pos % 2] = logical_v[pos]
    expected = torch.empty_like(q, dtype=torch.float32)
    for qi in range(query_len):
        visible = prefix + qi + 1
        for head in range(4):
            kh = logical_k[:visible, head // 2].float()
            vh = logical_v[:visible, head // 2].float()
            probabilities = (kh @ q[qi, head].float() / 2).softmax(0)
            expected[qi, head] = probabilities @ vh
    actual = paged_attention(
        q,
        k,
        v,
        torch.tensor([table], device=device),
        torch.tensor([context], device=device),
        torch.tensor([0, query_len], device=device),
    )
    assert actual.dtype == dtype
    assert actual.device == device
    assert torch.isfinite(actual).all()
    tolerance = 1e-5 if dtype == torch.float32 else 0.01
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)


def test_requests_cannot_attend_to_each_other_or_padding():
    q = torch.zeros(2, 1, 2)
    k = torch.zeros(3, 2, 1, 2)
    v = torch.full_like(k, 999)
    v[2, 0] = 2
    v[2, 1] = 4
    v[0, 0] = 10
    actual = paged_attention(
        q,
        k,
        v,
        torch.tensor([[2, -1], [0, -1]]),
        torch.tensor([2, 1]),
        torch.tensor([0, 1, 2]),
    )
    torch.testing.assert_close(actual, torch.tensor([[[3.0, 3.0]], [[10.0, 10.0]]]))


@pytest.mark.parametrize(
    "query_heads,context,table", [(3, 2, [0]), (4, 3, [0]), (4, 2, [-1])]
)
def test_invalid_head_ratio_or_cache_metadata_rejected(query_heads, context, table):
    with pytest.raises(ValueError):
        paged_attention(
            torch.zeros(1, query_heads, 4),
            torch.zeros(1, 2, 2, 4),
            torch.zeros(1, 2, 2, 4),
            torch.tensor([table]),
            torch.tensor([context]),
            torch.tensor([0, 1]),
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_accumulates_large_scores_without_half_precision_overflow(
    dtype, device
):
    # Identical logits should yield a uniform distribution, not inf/inf -> NaN.
    query = torch.full((1, 2, 4), 1000.0, dtype=dtype, device=device)
    keys = torch.full((1, 2, 1, 4), 1000.0, dtype=dtype, device=device)
    values = (
        torch.tensor([2.0, 4.0], dtype=dtype, device=device)
        .reshape(1, 2, 1, 1)
        .expand_as(keys)
    )
    actual = paged_attention(
        query,
        keys,
        values,
        torch.tensor([[0]], device=device),
        torch.tensor([2], device=device),
        torch.tensor([0, 1], device=device),
    )
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, torch.full_like(query, 3.0))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_small_block_paged_prefill_matches_reference(dtype):
    """Gather+varlen prefill must keep prefix reuse and physical block order."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("FlashAttention requires Ampere or newer")

    block_size, num_blocks = 16, 6
    num_q_heads, num_kv_heads, head_dim = 8, 2, 32
    contexts = [33, 17]
    q_lens = [16, 5]
    cu_q = torch.tensor([0, 16, 21], dtype=torch.int32, device="cuda")
    block_tables = torch.tensor(
        [[5, 1, 3], [0, 4, 2]], dtype=torch.int32, device="cuda"
    )
    keys = torch.full(
        (num_blocks, block_size, num_kv_heads, head_dim),
        1000.0,
        dtype=dtype,
        device="cuda",
    )
    values = torch.full_like(keys, -1000.0)
    logical_k = torch.randn(
        sum(contexts), num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    logical_v = torch.randn_like(logical_k)
    position = 0
    for request, context_len in enumerate(contexts):
        for token in range(context_len):
            block = block_tables[request, token // block_size]
            keys[block, token % block_size] = logical_k[position + token]
            values[block, token % block_size] = logical_v[position + token]
        position += context_len

    query = torch.randn(sum(q_lens), num_q_heads, head_dim, dtype=dtype, device="cuda")
    scale = 1.0 / head_dim**0.5
    expected = _paged_attention_reference(
        query,
        keys,
        values,
        block_tables,
        contexts,
        cu_q.tolist(),
        scale=scale,
    )
    actual = paged_attention(
        query,
        keys,
        values,
        block_tables,
        torch.tensor(contexts, dtype=torch.int32, device="cuda"),
        cu_q,
        scale=scale,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
