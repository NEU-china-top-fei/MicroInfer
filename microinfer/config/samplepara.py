import math
from dataclasses import dataclass


@dataclass(slots=True)
class SampleParameter:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    greedy: bool = False

    def __post_init__(self):
        assert math.isfinite(self.temperature) and self.temperature > 1e-10, (
            "greedy sampling is not permitted"
        )
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
