"""AttentionMetadata contracts: lazy CPU lists and slot indices."""

import pytest
import torch

from microinfer.layer.metadata import AttentionMetadata

pytestmark = pytest.mark.component


def test_prefill_metadata_derives_lists_lazily():
    query = torch.tensor([0, 2, 5], dtype=torch.int32)
    key = torch.tensor([0, 2, 5], dtype=torch.int32)
    metadata = AttentionMetadata.build(
        is_prefill=True,
        cu_seqlens_q=query,
        cu_seqlens_k=key,
        max_seqlen_q=3,
        max_seqlen_k=5,
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    assert metadata.direct_prefill
    assert metadata.cu_q_cpu == (0, 2, 5)
    assert metadata.q_lens_cpu == (2, 3)
    assert metadata.context_lens_cpu == (2, 3)
    # The tuple is cached; repeated attention layers do not re-run tolist().
    assert metadata.cu_q_cpu is metadata.cu_q_cpu


def test_decode_metadata_derives_query_offsets_from_context_lens():
    metadata = AttentionMetadata.build(
        is_prefill=False,
        context_lens=torch.tensor([3, 4], dtype=torch.int32),
        slot_mapping=torch.tensor([2, 3], dtype=torch.int32),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
    )
    assert metadata.is_decode
    assert not metadata.direct_prefill
    assert metadata.cu_q_cpu == (0, 1, 2)
    assert metadata.q_lens_cpu == (1, 1)
    assert metadata.context_lens_cpu == (3, 4)


def test_slot_indices_are_cached_per_block_size():
    metadata = AttentionMetadata.build(
        is_prefill=False,
        slot_mapping=torch.tensor([0, 5, 2], dtype=torch.int32),
    )
    blocks, offsets = metadata.slot_indices(4)
    assert blocks.tolist() == [0, 1, 0]
    assert offsets.tolist() == [0, 1, 2]
    assert metadata.slot_indices(4)[0] is blocks
    assert metadata.slot_mapping_long.dtype == torch.long

    blocks2, offsets2 = metadata.slot_indices(2)
    assert blocks2.tolist() == [0, 2, 1]
    assert offsets2.tolist() == [0, 1, 0]
