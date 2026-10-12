"""Optional numerical-test adapter; nano-vllm Runner uses Context directly."""

from dataclasses import dataclass

import torch

from microinfer.engine.contracts import SchedulerOutput


@dataclass
class ModelInput:
    input_ids: torch.Tensor  # int64 [T]
    positions: torch.Tensor  # int64 [T], absolute within each request
    slot_mapping: torch.Tensor  # int64 [T], physical_block * B + offset
    cu_seqlens_q: torch.Tensor  # int32 [N+1]
    context_lens: torch.Tensor  # int32 [N], prefix + this chunk
    block_tables: torch.Tensor  # int32 [N, max_blocks], padding = -1
    sample_indices: torch.Tensor  # int64 [S], last query row for sampling
    temperatures: torch.Tensor  # float32 [S]
    seq_ids: tuple[int, ...]  # N request IDs in packed order
    sample_seq_ids: tuple[int, ...]  # S IDs, same order as sample_indices


def prepare_batch(
    output: SchedulerOutput, block_size: int, device: str | torch.device = "cpu"
) -> ModelInput:
    """Do not mutate the snapshot. Reject invalid/duplicate IDs, empty items,
    negative slots/positions and insufficient block tables with ValueError.
    An empty SchedulerOutput is valid and has cu_seqlens_q=[0].
    """
    raise NotImplementedError("M02: prepare_batch")
