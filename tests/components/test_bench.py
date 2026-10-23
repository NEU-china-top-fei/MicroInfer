"""Pure-Python contract tests for the benchmark harness."""

import pytest

from bench.ablation import row_from_result
from bench.metrics import RequestTiming, describe, percentile
from bench.profile_runner import nsys_command
from bench.workloads import WORKLOAD_NAMES, build_workload

pytestmark = pytest.mark.component


def test_nsys_command_is_viewable_and_reenters_as_module():
    command = nsys_command(
        "nsys",
        "/tmp/report",
        "cuda,nvtx",
        ["-m", "bench.profile_runner", "--export", "run"],
        python_exe="/usr/bin/python",
    )
    assert command[0:2] == ["nsys", "profile"]
    assert "--trace=cuda,nvtx" in command
    assert "--output=/tmp/report" in command
    assert "--stats=false" in command
    assert command[-5:] == [
        "/usr/bin/python",
        "-m",
        "bench.profile_runner",
        "--export",
        "run",
    ]


def test_workloads_are_deterministic_and_well_formed():
    for name in WORKLOAD_NAMES:
        first = build_workload(name, seed=7)
        second = build_workload(name, seed=7)
        assert first == second
        assert first.num_requests == len(first.prompts)
        assert first.prompt_tokens == sum(len(p) for p in first.prompts)
        assert first.output_tokens == first.num_requests * first.output_len
        assert first.output_len > 0
        assert all(prompt for prompt in first.prompts)


def test_shared_prefix_workload_reuses_a_prefix():
    workload = build_workload("shared-prefix", seed=0)
    prefixes = {prompt[: workload.shared_prefix] for prompt in workload.prompts}
    assert workload.shared_prefix > 0
    assert len(prefixes) == 1


def test_percentile_interpolates_and_handles_empty():
    assert percentile([], 50) is None
    assert percentile([1.0], 99) == 1.0
    assert percentile([0.0, 10.0], 50) == pytest.approx(5.0)
    assert percentile([0.0, 10.0], 25) == pytest.approx(2.5)


def test_describe_empty_and_nonempty():
    empty = describe([])
    assert empty["count"] == 0 and empty["p50"] is None

    summary = describe([1.0, 2.0, 3.0])
    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["p50"] == pytest.approx(2.0)
    assert summary["min"] == 1.0 and summary["max"] == 3.0


def test_request_timing_derives_ttft_tpot_and_itl():
    timing = RequestTiming(4, (0.1, 0.3, 0.5))
    assert timing.output_len == 3
    assert timing.ttft_s == pytest.approx(0.1)
    assert timing.tpot_s == pytest.approx(0.2)
    assert timing.itls_s == pytest.approx([0.2, 0.2])


def test_request_timing_rejects_non_monotonic_arrivals():
    with pytest.raises(ValueError):
        RequestTiming(4, (0.2, 0.1))
    with pytest.raises(ValueError):
        RequestTiming(4, ())


def test_ablation_row_extracts_metrics():
    data = {
        "meta": {
            "engine": "microinfer",
            "cache_mode": "cold",
            "debug_mode": True,
            "use_compile": False,
        },
        "workload": {"name": "single-short"},
        "summary": {
            "requests": 3,
            "output_tps": {"p50": 10.0},
            "input_tps": {"p50": 20.0},
            "ttft_ms": {"p50": 1.0, "p99": 2.0},
            "tpot_ms": {"p50": 3.0},
            "peak_alloc_mb": 100.0,
            "prefix_cache_hit_rate": 0.5,
            "kv_utilization_peak": 0.25,
        },
    }
    row = row_from_result(
        data, "p1.0", timestamp="2026-01-01T00:00:00+00:00"
    )
    assert row["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert row["label"] == "p1.0"
    assert row["output_tps"] == 10.0
    assert row["ttft_p99_ms"] == 2.0
    assert row["peak_alloc_mb"] == 100.0
    assert row["debug_mode"] is True
    assert row["prefix_hit_rate"] == 0.5
