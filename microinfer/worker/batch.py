"""Optional numerical-test adapter; nano-vllm Runner uses Context directly."""

import math
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
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be positive")
    ids, positions, slots, offsets, lengths, samples, temperatures = (
        [],
        [],
        [],
        [0],
        [],
        [],
        [],
    )
    seq_ids, sample_ids, tables = [], [], []
    for item in output.items:
        if (
            type(item.seq_id) is not int
            or item.seq_id < 0
            or item.seq_id in seq_ids
            or type(item.start_pos) is not int
            or item.start_pos < 0
            or not item.token_ids
            or not math.isfinite(item.temperature)
            or item.temperature <= 0
            or any(type(t) is not int or t < 0 for t in item.token_ids)
            or any(type(b) is not int or b < 0 for b in item.block_table)
            or len(item.block_table) * block_size < item.context_len
        ):
            raise ValueError("invalid scheduled sequence metadata")
        seq_ids.append(item.seq_id)
        ids.extend(item.token_ids)
        current_positions = range(item.start_pos, item.context_len)
        positions.extend(current_positions)
        slots.extend(
            item.block_table[p // block_size] * block_size + p % block_size
            for p in current_positions
        )
        offsets.append(len(ids))
        lengths.append(item.context_len)
        tables.append(item.block_table)
        if item.sample:
            samples.append(len(ids) - 1)
            sample_ids.append(item.seq_id)
            temperatures.append(item.temperature)
    width = max(map(len, tables), default=0)

    def tensor(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device=device)

    blocks = tensor(
        [list(t) + [-1] * (width - len(t)) for t in tables], torch.int32
    ).reshape(len(tables), width)
    return ModelInput(
        tensor(ids),
        tensor(positions),
        tensor(slots),
        tensor(offsets, torch.int32),
        tensor(lengths, torch.int32),
        blocks,
        tensor(samples),
        tensor(temperatures, torch.float32),
        tuple(seq_ids),
        tuple(sample_ids),
    )
