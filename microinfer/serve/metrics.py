"""Dependency-free Prometheus text metrics for the OpenAI-compatible server."""

from __future__ import annotations


class Histogram:
    def __init__(self, buckets):
        self.buckets = tuple(sorted(float(b) for b in buckets))
        self.counts = [0] * len(self.buckets)
        self.count = 0
        self.total = 0.0

    def observe(self, value: float):
        value = max(0.0, float(value))
        self.count += 1
        self.total += value
        for index, upper in enumerate(self.buckets):
            if value <= upper:
                self.counts[index] += 1

    def render(self, name: str, help_text: str) -> list[str]:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        for upper, count in zip(self.buckets, self.counts):
            lines.append(f'{name}_bucket{{le="{upper:g}"}} {count}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{name}_sum {self.total:.9f}")
        lines.append(f"{name}_count {self.count}")
        return lines


class ServiceMetrics:
    def __init__(self):
        self.counters = {
            "microinfer_requests_total": 0,
            "microinfer_requests_succeeded_total": 0,
            "microinfer_requests_failed_total": 0,
            "microinfer_requests_aborted_total": 0,
            "microinfer_stream_requests_total": 0,
            "microinfer_prompt_tokens_total": 0,
            "microinfer_completion_tokens_total": 0,
        }
        self.ttft = Histogram((0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5))
        self.tpot = Histogram((0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5))

    def inc(self, name: str, value: int = 1):
        if name not in self.counters:
            raise KeyError(name)
        self.counters[name] += value

    def render(self, engine_stats: dict) -> str:
        lines = []
        for name, value in self.counters.items():
            lines.extend([f"# TYPE {name} counter", f"{name} {value}"])
        lines.extend(
            self.ttft.render(
                "microinfer_ttft_seconds", "Time from admit to first completion token"
            )
        )
        lines.extend(
            self.tpot.render(
                "microinfer_tpot_seconds",
                "Time per output token after the first token",
            )
        )
        gauges = {
            "microinfer_running_requests": engine_stats.get("running", 0),
            "microinfer_waiting_requests": engine_stats.get("waiting", 0),
            "microinfer_active_requests": engine_stats.get("active_requests", 0),
            "microinfer_kv_used_blocks": engine_stats.get("used_blocks", 0),
            "microinfer_kv_free_blocks": engine_stats.get("free_blocks", 0),
            "microinfer_kv_utilization": engine_stats.get("kv_utilization", 0.0),
            "microinfer_prefix_cache_entries": engine_stats.get(
                "prefix_cache_entries", 0
            ),
            "microinfer_prefix_cache_hit_tokens": engine_stats.get(
                "prefix_cache_hit_tokens", 0
            ),
            "microinfer_prefix_cache_evictions": engine_stats.get(
                "prefix_cache_evictions", 0
            ),
            "microinfer_preemptions": engine_stats.get("num_preemptions", 0),
            "microinfer_steps": engine_stats.get("num_steps", 0),
            "microinfer_spec_steps": engine_stats.get("num_spec_steps", 0),
            "microinfer_spec_draft_tokens": engine_stats.get(
                "num_spec_draft_tokens", 0
            ),
            "microinfer_spec_accepted_tokens": engine_stats.get(
                "num_spec_accepted_tokens", 0
            ),
            "microinfer_spec_acceptance_rate": engine_stats.get(
                "spec_acceptance_rate"
            )
            or 0.0,
        }
        for name, value in gauges.items():
            lines.extend([f"# TYPE {name} gauge", f"{name} {value}"])
        return "\n".join(lines) + "\n"
