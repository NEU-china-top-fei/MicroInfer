import pytest
import torch

from microinfer.utils.context import get_context, reset_context, set_context

pytestmark = pytest.mark.nano


def test_context_reset_removes_all_request_metadata():
    slots = torch.tensor([2, 3], dtype=torch.int32)
    try:
        set_context(True, cu_seqlens_q=torch.tensor([0, 2]), slot_mapping=slots)
        assert get_context().is_prefill and get_context().slot_mapping is slots
        reset_context()
        context = get_context()
        assert not context.is_prefill and context.slot_mapping is None
        assert (
            context.cu_seqlens_q is context.cu_seqlens_k is context.block_tables is None
        )
    finally:
        reset_context()


def test_decode_replaces_prefill_fields_instead_of_leaking_them():
    try:
        set_context(True, cu_seqlens_q=torch.tensor([0, 2]), max_seqlen_q=2)
        set_context(False, context_lens=torch.tensor([3]))
        assert get_context().cu_seqlens_q is None and get_context().max_seqlen_q == 0
        assert get_context().context_lens.tolist() == [3]
    finally:
        reset_context()
