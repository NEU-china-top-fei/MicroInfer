"""Test-only scheduler snapshots (previously shipped in engine.contracts)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledSequence:
    seq_id: int
    token_ids: tuple[int, ...]
    start_pos: int
    block_table: tuple[int, ...]
    sample: bool
    temperature: float = 1.0

    @property
    def context_len(self) -> int:
        return self.start_pos + len(self.token_ids)


@dataclass(frozen=True)
class SchedulerOutput:
    items: tuple[ScheduledSequence, ...] = ()

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(len(item.token_ids) for item in self.items)
