import pytest
import torch

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
