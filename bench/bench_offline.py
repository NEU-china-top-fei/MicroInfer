"""Run one engine over deterministic offline workloads and write JSON results."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from bench.engines import build_engine
from bench.metrics import aggregate_runs
from bench.workloads import WORKLOAD_NAMES, build_workloads

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _meta(args, engine) -> dict:
    return {
        "schema_version": 1,
        "engine": args.engine,
        "tag": args.tag,
        "model_name": Path(args.model).name,
        "device": args.device,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else None
        ),
        "python": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "engine_meta": engine.meta,
        "cache_mode": args.cache_mode,
        "debug_mode": args.debug_mode if args.engine == "microinfer" else False,
        "use_compile": args.use_compile if args.engine == "microinfer" else False,
        "fuse_weights": (
            args.fuse_weights if args.engine == "microinfer" else False
        ),
        "use_cuda_graph": (
            args.use_cuda_graph if args.engine == "microinfer" else False
        ),
        "bench": {
            "runs": args.runs,
            "warmup": args.warmup,
            "seed": args.seed,
            "workloads": args.workloads,
        },
    }


def _print_header() -> None:
    print(
        f"{'workload':<18} {'out tok/s':>10} {'TTFT p50':>10} "
        f"{'TTFT p99':>10} {'TPOT p50':>10} {'peak MB':>9}"
    )


def _print_row(name: str, summary: dict) -> None:
    def value(section: str, key: str, digits: int = 1) -> str:
        found = summary[section].get(key)
        return "-" if found is None else f"{found:.{digits}f}"

    peak = summary.get("peak_alloc_mb")
    print(
        f"{name:<18} {value('output_tps', 'p50', 1):>10} "
        f"{value('ttft_ms', 'p50'):>10} {value('ttft_ms', 'p99'):>10} "
        f"{value('tpot_ms', 'p50'):>10} "
        f"{'-' if peak is None else f'{peak:.0f}':>9}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="local model directory")
    parser.add_argument(
        "--engine", choices=("microinfer", "hf", "nano"), default="microinfer"
    )
    parser.add_argument(
        "--workloads", default=",".join(WORKLOAD_NAMES), help="comma-separated"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--nano-root", default=None)
    parser.add_argument("--nano-enforce-eager", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "warm"),
        default="cold",
        help="cold: reset prefix cache per run; warm: steady-state repeated prompts",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="enable scheduler/runner invariant checks (MicroInfer only)",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="compile model forward (experimental; MicroInfer only)",
    )
    parser.add_argument(
        "--no-fuse-weights",
        action="store_false",
        dest="fuse_weights",
        help="keep separate QKV/gate-up weights (lower memory; MicroInfer only)",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="capture decode buckets with CUDA Graph (MicroInfer only)",
    )
    parser.add_argument("--output-dir", default="results/bench")
    parser.add_argument("--tag", default="")
    parser.add_argument("--list-workloads", action="store_true")
    args = parser.parse_args()

    if args.list_workloads:
        for name in WORKLOAD_NAMES:
            print(name)
        return 0
    if not args.model:
        parser.error("--model is required unless --list-workloads is used")

    names = [name.strip() for name in args.workloads.split(",") if name.strip()]
    if not names:
        parser.error("--workloads must not be empty")
    if args.runs <= 0 or args.warmup < 0:
        parser.error("--runs must be positive and --warmup nonnegative")

    if args.engine == "microinfer" and args.block_size % 256 != 0:
        print(
            f"note: block_size={args.block_size} disables the FlashAttention paged "
            "path in the current implementation"
        )

    engine = build_engine(
        args.engine,
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        attn_implementation=args.attn_implementation,
        nano_root=args.nano_root,
        enforce_eager=args.nano_enforce_eager,
        cache_mode=args.cache_mode,
        debug_mode=args.debug_mode,
        use_compile=args.use_compile,
        fuse_weights=args.fuse_weights,
        use_cuda_graph=args.use_cuda_graph,
    ).load()

    workloads = build_workloads(names, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{_slug(args.tag)}" if args.tag else ""
    if args.cache_mode == "warm":
        suffix += "_warm"

    meta = _meta(args, engine)
    print(
        f"engine={args.engine} model={meta['model_name']} gpu={meta['gpu']} "
        f"torch={meta['torch']}"
    )
    print(json.dumps(engine.meta, sort_keys=True))
    _print_header()

    for workload in workloads:
        runs = engine.run(
            workload, warmup=args.warmup, runs=args.runs, timeout_s=args.timeout
        )
        summary = aggregate_runs(runs)
        result = {
            "schema_version": 1,
            "meta": meta,
            "workload": workload.describe(),
            "runs": [run.to_dict() for run in runs],
            "summary": summary,
        }
        path = output_dir / f"{args.engine}_{workload.name}{suffix}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        _print_row(workload.name, summary)

    print(f"results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
