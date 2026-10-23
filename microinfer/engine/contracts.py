"""Output records shared by the engine and the text frontend."""

from dataclasses import dataclass
from typing import Literal


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
