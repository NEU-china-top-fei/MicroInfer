"""Optional CPU-only helper values for batch/reference tests and text processing.

The primary nano-vllm path passes (list[Sequence], is_prefill), NOT SchedulerOutput.
Snapshots remain available for isolated packing and cached-model oracle tests.
See docs/architecture.md for lifecycle and validation rules.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ScheduledSequence:
    seq_id: int
    token_ids: tuple[int, ...]  # Only this step's query tokens.
    start_pos: int  # Number of KV tokens already computed.
    block_table: tuple[int, ...]
    sample: bool  # This step reaches the end of known tokens.
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


@dataclass(frozen=True)
class RequestOutput:
    seq_id: int
    token_ids: tuple[int, ...]  # Cumulative completion, excludes prompt.
    finished: bool
    finish_reason: Literal["eos", "length"] | None = None


@dataclass(frozen=True)
class TextOutput:
    seq_id: int
    token_ids: tuple[int, ...]
    text: str
    finished: bool
    finish_reason: Literal["eos", "length"] | None = None
