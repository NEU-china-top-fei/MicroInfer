"""Deterministic offline workloads shared by every benchmark adapter."""

from __future__ import annotations

import random
from dataclasses import dataclass

# Keep random IDs away from special-token ranges. Correctness is irrelevant for
# throughput measurement; every workload generates exactly output_len tokens.
TOKEN_LOW = 100
TOKEN_HIGH = 20_000

WORKLOAD_NAMES = (
    "single-short",
    "batch8-short",
    "batch32-mixed",
    "shared-prefix",
    "long-prefill",
)


@dataclass(frozen=True)
class Workload:
    name: str
    prompts: tuple[tuple[int, ...], ...]
    output_len: int
    seed: int
    shared_prefix: int = 0

    @property
    def num_requests(self) -> int:
        return len(self.prompts)

    @property
    def prompt_tokens(self) -> int:
        return sum(len(p) for p in self.prompts)

    @property
    def output_tokens(self) -> int:
        return self.num_requests * self.output_len

    def describe(self) -> dict:
        lengths = [len(p) for p in self.prompts]
        return {
            "name": self.name,
            "num_requests": self.num_requests,
            "prompt_tokens": self.prompt_tokens,
            "prompt_len_min": min(lengths),
            "prompt_len_max": max(lengths),
            "prompt_len_mean": round(self.prompt_tokens / self.num_requests, 2),
            "output_len": self.output_len,
            "output_tokens": self.output_tokens,
            "shared_prefix": self.shared_prefix,
            "seed": self.seed,
        }


def _tokens(rng: random.Random, count: int) -> tuple[int, ...]:
    return tuple(rng.randrange(TOKEN_LOW, TOKEN_HIGH) for _ in range(count))


def build_workload(name: str, seed: int = 0) -> Workload:
    """Build a workload from a stable seed.

    ``random.Random(str)`` is deterministic across processes and Python
    versions, unlike ``hash()``.
    """
    if name not in WORKLOAD_NAMES:
        raise ValueError(f"unknown workload {name!r}; choose from {WORKLOAD_NAMES}")
    rng = random.Random(f"{seed}:{name}")

    if name == "single-short":
        prompts = (_tokens(rng, 128),)
        output_len = 32
    elif name == "batch8-short":
        prompts = tuple(_tokens(rng, 128) for _ in range(8))
        output_len = 32
    elif name == "batch32-mixed":
        prompts = tuple(_tokens(rng, rng.randint(64, 384)) for _ in range(32))
        output_len = 32
    elif name == "shared-prefix":
        prefix = _tokens(rng, 512)
        prompts = tuple(prefix + _tokens(rng, 32) for _ in range(8))
        output_len = 32
        return Workload(name, prompts, output_len, seed, shared_prefix=512)
    elif name == "long-prefill":
        prompts = (_tokens(rng, 2048),)
        output_len = 32
    else:  # pragma: no cover - guarded above
        raise AssertionError(name)

    return Workload(name, prompts, output_len, seed)


def build_workloads(names: list[str], seed: int = 0) -> list[Workload]:
    return [build_workload(name, seed) for name in names]
