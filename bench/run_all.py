"""Run every available engine in its own process, then render the comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--engines", default="microinfer,hf,nano", help="comma-separated"
    )
    parser.add_argument("--workloads", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--nano-root", default=None)
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "warm"),
        default="cold",
        help="cold resets prefix cache per run; warm measures steady state",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="enable MicroInfer scheduler/runner invariant checks",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="compile MicroInfer model forward (experimental)",
    )
    parser.add_argument(
        "--no-fuse-weights",
        action="store_false",
        dest="fuse_weights",
        help="keep separate QKV/gate-up weights (lower memory)",
    )
    parser.add_argument("--output-dir", default="results/bench")
    parser.add_argument("--skip-bs16", action="store_true")
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="retry a failed engine process (transient CUDA/driver aborts)",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries must be nonnegative")

    engines = [name.strip() for name in args.engines.split(",") if name.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        "-m",
        "bench.bench_offline",
        "--model",
        args.model,
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--cache-mode",
        args.cache_mode,
        "--output-dir",
        str(output_dir),
    ]
    if args.workloads:
        common += ["--workloads", args.workloads]

    commands: list[tuple[str, list[str]]] = []
    if "microinfer" in engines:
        for block_size, num_blocks, tag in ((256, 64, "bs256"), (16, 1024, "bs16")):
            if tag == "bs16" and args.skip_bs16:
                continue
            command = common + [
                "--engine",
                "microinfer",
                "--block-size",
                str(block_size),
                "--num-blocks",
                str(num_blocks),
                "--tag",
                tag,
            ]
            if args.debug_mode:
                command.append("--debug-mode")
            if args.use_compile:
                command.append("--use-compile")
            if not args.fuse_weights:
                command.append("--no-fuse-weights")
            commands.append((f"microinfer {tag}", command))
    if "hf" in engines:
        commands.append(("hf", common + ["--engine", "hf"]))
    if "nano" in engines:
        nano = common + ["--engine", "nano"]
        if args.nano_root:
            nano += ["--nano-root", args.nano_root]
        commands.append(("nano", nano))

    failures = []
    for label, command in commands:
        result = None
        for attempt in range(args.retries + 1):
            suffix = f" (attempt {attempt + 1})" if args.retries else ""
            print(f"\n=== {label}{suffix} ===", flush=True)
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if result.returncode == 0:
                break
            print(f"{label} failed with exit code {result.returncode}", flush=True)
        if result is None or result.returncode != 0:
            failures.append(label)

    report = [
        sys.executable,
        "-m",
        "bench.report",
        "--input-dir",
        str(output_dir),
        "--output",
        str(output_dir / "summary.md"),
        "--cache-mode",
        args.cache_mode,
    ]
    subprocess.run(report, cwd=PROJECT_ROOT, check=False)

    if failures:
        print(f"\nfailed: {', '.join(failures)}")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
