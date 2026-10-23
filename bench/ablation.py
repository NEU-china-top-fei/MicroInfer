"""Run one labelled benchmark and append its rows to a CSV ablation log.

This is the P1 safety net: every optimization step gets a label, a fixed
workload set, and a comparable row. Keep the machine and cold/warm cache mode
constant across labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "timestamp",
    "label",
    "engine",
    "cache_mode",
    "debug_mode",
    "use_compile",
    "workload",
    "requests",
    "output_tps",
    "input_tps",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "peak_alloc_mb",
    "prefix_hit_rate",
    "kv_utilization_peak",
]


def _summary_value(summary: dict, section: str, key: str):
    value = summary.get(section)
    if isinstance(value, dict):
        return value.get(key)
    return value


def row_from_result(data: dict, label: str, *, timestamp: str | None = None) -> dict:
    summary = data["summary"]
    meta = data["meta"]
    return {
        "timestamp": timestamp
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "engine": meta["engine"],
        "cache_mode": meta.get("cache_mode", "cold"),
        "debug_mode": meta.get("debug_mode", False),
        "use_compile": meta.get("use_compile", False),
        "workload": data["workload"]["name"],
        "requests": summary.get("requests"),
        "output_tps": _summary_value(summary, "output_tps", "p50"),
        "input_tps": _summary_value(summary, "input_tps", "p50"),
        "ttft_p50_ms": _summary_value(summary, "ttft_ms", "p50"),
        "ttft_p99_ms": _summary_value(summary, "ttft_ms", "p99"),
        "tpot_p50_ms": _summary_value(summary, "tpot_ms", "p50"),
        "peak_alloc_mb": summary.get("peak_alloc_mb"),
        "prefix_hit_rate": summary.get("prefix_cache_hit_rate"),
        "kv_utilization_peak": summary.get("kv_utilization_peak"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, help="e.g. p1.1-metadata")
    parser.add_argument(
        "--workloads",
        default="single-short,batch8-short,long-prefill,shared-prefix",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--cache-mode", choices=("cold", "warm"), default="cold")
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument(
        "--no-fuse-weights",
        action="store_false",
        dest="fuse_weights",
        help="keep separate QKV/gate-up weights (lower memory)",
    )
    parser.add_argument("--csv", default="results/bench/ablation.csv")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="keep raw JSON here instead of a temporary directory",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    temporary = None
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="microinfer-ablation-")
        raw_dir = Path(temporary.name)

    command = [
        sys.executable,
        "-m",
        "bench.bench_offline",
        "--model",
        args.model,
        "--engine",
        "microinfer",
        "--workloads",
        args.workloads,
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--block-size",
        str(args.block_size),
        "--num-blocks",
        str(args.num_blocks),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--cache-mode",
        args.cache_mode,
        "--output-dir",
        str(raw_dir),
        "--tag",
        args.label,
    ]
    if args.debug_mode:
        command.append("--debug-mode")
    if args.use_compile:
        command.append("--use-compile")
    if not args.fuse_weights:
        command.append("--no-fuse-weights")

    print("command:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        return result.returncode

    files = sorted(raw_dir.glob("*.json"))
    if not files:
        print(f"no JSON results in {raw_dir}", file=sys.stderr)
        return 1
    rows = []
    for path in files:
        rows.append(row_from_result(json.loads(path.read_text()), args.label, timestamp=timestamp))
    rows.sort(key=lambda row: row["workload"])

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    def fmt(row, key, digits=1):
        value = row[key]
        return "-" if value is None else f"{value:.{digits}f}"

    print()
    print(
        "| label | workload | out tok/s | TTFT p50 ms | TTFT p99 ms | "
        "TPOT p50 ms | peak MB | prefix hit |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        hit = row["prefix_hit_rate"]
        hit = "-" if hit is None else f"{hit * 100:.1f}%"
        print(
            f"| {row['label']} | {row['workload']} | {fmt(row, 'output_tps')} | "
            f"{fmt(row, 'ttft_p50_ms')} | {fmt(row, 'ttft_p99_ms')} | "
            f"{fmt(row, 'tpot_p50_ms')} | {fmt(row, 'peak_alloc_mb', 0)} | {hit} |"
        )
    print(f"\nablation log: {csv_path}")
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
