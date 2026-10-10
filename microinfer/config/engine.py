"""Single-process engine limits, independent of model dimensions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 512
    max_model_len: int = 4096
    eos_token_id: int | None = None

    def __post_init__(self):
        for name in ("max_num_seqs", "max_num_batched_tokens", "max_model_len"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.eos_token_id is not None and (
            type(self.eos_token_id) is not int or self.eos_token_id < 0
        ):
            raise ValueError("eos_token_id must be a nonnegative integer or None")
