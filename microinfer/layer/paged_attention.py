"""M04 exercise: CPU reference first, optional optimized CUDA backend later."""

import torch


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal GQA over paged KV that already includes this step's new K/V.

    Q=[total_queries,Hq,D], K/V=[physical_blocks,B,Hkv,D]. Return Q-shaped output.
    Hq must be divisible by Hkv. Query j in a request can see keys through
    context_len - query_len + j (inclusive), not just through j. Ignore padded
    block IDs beyond context_len. Invalid metadata must raise ValueError.
    """
    raise NotImplementedError("M04: paged_attention")
