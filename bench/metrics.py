"""Timing records and aggregation shared by benchmark adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def percentile(values, p: float) -> float | None:
    """Linear-interpolated percentile; ``None`` for an empty sample."""
    if not 0 <= p <= 100:
        raise ValueError("percentile must be within [0, 100]")
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def describe(values) -> dict:
    """Summary used in JSON and markdown reports."""
    imported = [float(v) for v in values]
    if not imported:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(imported),
        "mean": sum(imported) / len(imported),
        "p50": percentile(imported, 50),
        "p90": percentile(imported, 90),
        "p99": percentile(imported, 99),
        "min": min(imported),
        "max": max(imported),
    }


@dataclass(frozen=True)
class RequestTiming:
    """Token arrival times for one request, relative to the run start."""

    prompt_len: int
    token_times_s: tuple[float, ...]

    def __post_init__(self):
        if self.prompt_len <= 0:
            raise ValueError("prompt_len must be positive")
        if not self.token_times_s:
            raise ValueError("request produced no tokens")
        previous = -1.0
        for t in self.token_times_s:
            if t < 0 or t < previous:
                raise ValueError("token times must be nondecreasing and nonnegative")
            previous = t

    @property
    def output_len(self) -> int:
        return len(self.token_times_s)

    @property
    def ttft_s(self) -> float:
        return self.token_times_s[0]

    @property
    def tpot_s(self) -> float:
        if self.output_len == 1:
            return 0.0
        return (self.token_times_s[-1] - self.token_times_s[0]) / (
            self.output_len - 1
        )

    @property
    def itls_s(self) -> list[float]:
        return [
            b - a for a, b in zip(self.token_times_s, self.token_times_s[1:])
        ]


@dataclass(frozen=True)
class RunResult:
    wall_s: float
    timings: tuple[RequestTiming, ...]
    steps: int
    peak_alloc_mb: float | None
    peak_reserved_mb: float | None
    extra: dict = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_len for t in self.timings)

    @property
    def output_tokens(self) -> int:
        return sum(t.output_len for t in self.timings)

    @property
    def output_tps(self) -> float:
        return self.output_tokens / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def input_tps(self) -> float:
        return self.prompt_tokens / self.wall_s if self.wall_s > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "wall_s": self.wall_s,
            "steps": self.steps,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "output_tps": self.output_tps,
            "input_tps": self.input_tps,
            "peak_alloc_mb": self.peak_alloc_mb,
            "peak_reserved_mb": self.peak_reserved_mb,
            "ttft_ms": describe([t.ttft_s * 1000 for t in self.timings]),
            "tpot_ms": describe([t.tpot_s * 1000 for t in self.timings]),
            "itl_ms": describe([x * 1000 for t in self.timings for x in t.itls_s]),
            "extra": self.extra,
        }


def aggregate_runs(runs: list[RunResult]) -> dict:
    """Pool request latency across runs; throughput is per-run then summarized."""
    if not runs:
        raise ValueError("at least one run is required")
    ttfts = [t.ttft_s * 1000 for r in runs for t in r.timings]
    tpots = [t.tpot_s * 1000 for r in runs for t in r.timings]
    itls = [x * 1000 for r in runs for t in r.timings for x in t.itls_s]
    alloc = [r.peak_alloc_mb for r in runs if r.peak_alloc_mb is not None]
    reserved = [r.peak_reserved_mb for r in runs if r.peak_reserved_mb is not None]
    kv = [
        r.extra["kv_utilization_peak"]
        for r in runs
        if r.extra.get("kv_utilization_peak") is not None
    ]
    hit_tokens = [
        r.extra["prefix_cache_hit_tokens"]
        for r in runs
        if r.extra.get("prefix_cache_hit_tokens") is not None
    ]
    prompt_tokens = sum(r.prompt_tokens for r in runs)
    return {
        "requests": sum(len(r.timings) for r in runs),
        "prompt_tokens": sum(r.prompt_tokens for r in runs),
        "output_tokens": sum(r.output_tokens for r in runs),
        "wall_s": describe([r.wall_s for r in runs]),
        "output_tps": describe([r.output_tps for r in runs]),
        "input_tps": describe([r.input_tps for r in runs]),
        "ttft_ms": describe(ttfts),
        "tpot_ms": describe(tpots),
        "itl_ms": describe(itls),
        "steps": describe([r.steps for r in runs]),
        "peak_alloc_mb": max(alloc) if alloc else None,
        "peak_reserved_mb": max(reserved) if reserved else None,
        "kv_utilization_peak": max(kv) if kv else None,
        "prefix_cache_hit_tokens": sum(hit_tokens) if hit_tokens else None,
        "prefix_cache_hit_rate": (
            sum(hit_tokens) / prompt_tokens if hit_tokens and prompt_tokens else None
        ),
    }
