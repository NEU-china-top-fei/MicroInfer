import pytest
import torch

from microinfer.layer.paged_attention import paged_attention

pytestmark = pytest.mark.roadmap


@pytest.mark.parametrize("prefix,query_len", [(0, 3), (2, 1), (1, 2)])
def test_paged_causal_gqa_matches_dense_reference(
    prefix, query_len, seeded_torch, device
):
    # Physical order 2 -> 0 differs from logical order; trailing slots are poison.
    context = prefix + query_len
    q = torch.randn(query_len, 4, 4).to(device)
    logical_k = torch.randn(context, 2, 4).to(device)
    logical_v = torch.randn(context, 2, 4).to(device)
    k = torch.full((3, 2, 2, 4), 1000.0, device=device)
    v = torch.full_like(k, -1000)
    table = [2, 0]
    for pos in range(context):
        k[table[pos // 2], pos % 2] = logical_k[pos]
        v[table[pos // 2], pos % 2] = logical_v[pos]
    expected = torch.empty_like(q)
    for qi in range(query_len):
        visible = prefix + qi + 1
        for head in range(4):
            kh = logical_k[:visible, head // 2]
            vh = logical_v[:visible, head // 2]
            probabilities = (kh @ q[qi, head] / 2).softmax(0)
            expected[qi, head] = probabilities @ vh
    actual = paged_attention(
        q,
        k,
        v,
        torch.tensor([table], device=device),
        torch.tensor([context], device=device),
        torch.tensor([0, query_len], device=device),
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


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
