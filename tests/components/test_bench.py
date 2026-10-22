"""Pure-Python contract tests for the benchmark harness."""

import pytest

from bench.metrics import RequestTiming, describe, percentile
from bench.workloads import WORKLOAD_NAMES, build_workload

pytestmark = pytest.mark.component


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
