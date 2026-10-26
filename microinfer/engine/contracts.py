"""Output records shared by the engine and the text frontend."""

from dataclasses import dataclass
from typing import Literal

FinishReason = Literal["eos", "length", "abort", "stop"]


@dataclass(frozen=True)
class RequestOutput:
    seq_id: int
    token_ids: tuple[int, ...]  # Cumulative completion, excludes prompt.
    finished: bool
    finish_reason: FinishReason | None = None


@dataclass(frozen=True)
class TextOutput:
    seq_id: int
    token_ids: tuple[int, ...]
    text: str
    finished: bool
    finish_reason: FinishReason | None = None


@dataclass(frozen=True)
class StreamEvent:
    """Incremental output for one request; ``text_delta`` is new text only."""

    seq_id: int
    token_ids: tuple[int, ...]
    text_delta: str
    finished: bool
    finish_reason: FinishReason | None = None
