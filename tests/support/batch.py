"""Test-only packed-batch adapter (previously shipped in microinfer.worker.batch)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tests.support.contracts import SchedulerOutput


@dataclass
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    cu_seqlens_q: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    sample_indices: torch.Tensor
    temperatures: torch.Tensor
    seq_ids: tuple[int, ...]
    sample_seq_ids: tuple[int, ...]


def prepare_batch(
    output: SchedulerOutput, block_size: int, device: str | torch.device = "cpu"
) -> ModelInput:
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
