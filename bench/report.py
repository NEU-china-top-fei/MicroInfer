"""Render benchmark JSON files as a markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.workloads import WORKLOAD_NAMES

ENGINE_ORDER = {"microinfer": 0, "hf": 1, "nano": 2}


def _load(paths: list[Path]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for path in paths:
        data = json.loads(path.read_text())
        workload = data["workload"]["name"]
        label = data["meta"]["engine"]
        tag = data["meta"].get("tag")
        if tag:
            label = f"{label} ({tag})"
        key = (workload, label)
        previous = latest.get(key)
        if previous is None or data["meta"]["timestamp"] > previous["meta"]["timestamp"]:
            latest[key] = data
    return list(latest.values())


def _label(data: dict) -> str:
    tag = data["meta"].get("tag")
    label = data["meta"]["engine"] + (f" ({tag})" if tag else "")
    if data["meta"].get("cache_mode", "cold") == "warm":
        label += " warm"
    return label


def _value(data: dict, section: str, key: str, digits: int = 1) -> str:
    found = data["summary"].get(section)
    if not isinstance(found, dict):
        return "-" if found is None else f"{found:.{digits}f}"
    value = found.get(key)
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/bench")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--baseline",
        default=None,
        help="engine name for the speedup column; defaults to hf, then nano",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "warm", "all"),
        default="cold",
        help="filter results by cache mode",
    )
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob("*.json"))
    if not paths:
        raise SystemExit(f"no JSON results in {args.input_dir}")
    data = _load(paths)
    if args.cache_mode != "all":
        data = [
            item
            for item in data
            if item["meta"].get("cache_mode", "cold") == args.cache_mode
        ]
    if not data:
        raise SystemExit(f"no {args.cache_mode} results in {args.input_dir}")

    engines = {item["meta"]["engine"] for item in data}
    baseline = args.baseline
    if baseline is None:
        baseline = "hf" if "hf" in engines else ("nano" if "nano" in engines else None)

    baseline_tps = {
        item["workload"]["name"]: item["summary"]["output_tps"]["p50"]
        for item in data
        if item["meta"]["engine"] == baseline
    }

    rows = []
    for item in data:
        workload = item["workload"]["name"]
        tps = item["summary"]["output_tps"]["p50"]
        speedup = "-"
        if workload in baseline_tps and item["meta"]["engine"] != baseline and tps:
            speedup = f"{tps / baseline_tps[workload]:.2f}x"
        rows.append(
            (
                WORKLOAD_NAMES.index(workload) if workload in WORKLOAD_NAMES else 99,
                ENGINE_ORDER.get(item["meta"]["engine"], 9),
                _label(item),
                workload,
                _value(item, "output_tps", "p50"),
                _value(item, "ttft_ms", "p50"),
                _value(item, "ttft_ms", "p99"),
                _value(item, "tpot_ms", "p50"),
                _value(item, "peak_alloc_mb", "max", 0),
                speedup,
            )
        )
    rows.sort(key=lambda row: (row[0], row[1], row[2]))

    lines = [
        (
            "| workload | engine | out tok/s | TTFT p50 ms | TTFT p99 ms | "
            "TPOT p50 ms | peak MB | throughput vs baseline |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, _, label, workload, tps, ttft50, ttft99, tpot50, peak, speedup in rows:
        lines.append(
            f"| {workload} | {label} | {tps} | {ttft50} | {ttft99} | "
            f"{tpot50} | {peak} | {speedup} |"
        )
    table = "\n".join(lines)
    print(table)

    cache_lines = [
        "",
        "| workload | engine | KV peak | prefix hit tokens | prefix hit rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in sorted(data, key=lambda x: x["workload"]["name"]):
        summary = item["summary"]
        kv = summary.get("kv_utilization_peak")
        if kv is None:
            continue
        hits = summary.get("prefix_cache_hit_tokens")
        rate = summary.get("prefix_cache_hit_rate")
        cache_lines.append(
            f"| {item['workload']['name']} | {_label(item)} | {kv * 100:.1f}% | "
            f"{'-' if hits is None else hits} | "
            f"{'-' if rate is None else f'{rate * 100:.1f}%'} |"
        )
    if len(cache_lines) > 2:
        print("\n".join(cache_lines))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n" + "\n".join(cache_lines) + "\n")
        print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
