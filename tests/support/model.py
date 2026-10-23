"""Test-only helpers for cached model forwards (previously QwenModel.forward_batch)."""

from __future__ import annotations

import torch

from microinfer.layer.metadata import AttentionMetadata


def forward_batch(model, batch):
    """All-query logits; the test must bind the cache on a ModelRunner first."""
    lengths = batch.context_lens
    cu_k = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(torch.int32)))
    q_lengths = batch.cu_seqlens_q[1:] - batch.cu_seqlens_q[:-1]
    metadata = AttentionMetadata.build(
        is_prefill=True,
        cu_seqlens_q=batch.cu_seqlens_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=int(q_lengths.max()) if q_lengths.numel() else 0,
        max_seqlen_k=int(lengths.max()) if lengths.numel() else 0,
        slot_mapping=batch.slot_mapping,
        block_tables=batch.block_tables,
    )
    return model.out_logits(model(batch.input_ids, batch.positions, metadata))
