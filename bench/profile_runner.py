"""Capture torch.profiler, TensorBoard, or Nsight Systems traces for one workload.

Examples:
    python -m bench.profile_runner --model ... --workload single-short
    python -m bench.profile_runner --model ... --export tensorboard
    python -m bench.profile_runner --model ... --workload long-prefill --max-tokens 1 --cold --export nsys

`--export torch` writes both a Chrome/Perfetto JSON and a TensorBoard trace.
TensorBoard's PyTorch Profiler plugin opens ``*.pt.trace.json.gz``; Perfetto and
``chrome://tracing`` open the plain ``*.json``. nsys must run the workload in a
child process because CUPTI cannot be shared with torch.profiler.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import os
import shlex
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from bench.workloads import WORKLOAD_NAMES, build_workload

EXPORT_CHOICES = ("torch", "chrome", "tensorboard", "nsys", "run")


def nsys_command(
    nsys_bin: str,
    output_base: str | Path,
    trace: str,
    child_argv: list[str],
    *,
    python_exe: str | None = None,
) -> list[str]:
    """Build a reproducible ``nsys profile`` command."""
    return [
        nsys_bin,
        "profile",
        f"--trace={trace}",
        "--sample=none",
        "--cpuctxsw=none",
        # Keep the report small and avoid blocking on CUDA symbol downloads.
        "--resolve-symbols=false",
        "--wait=primary",
        # Avoid the post-collection symbol download step (it can block offline).
        "--stats=false",
        "--force-overwrite=true",
        f"--output={output_base}",
        python_exe or sys.executable,
        *child_argv,
    ]


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_model(args, workload, *, nvtx: bool):
    from microinfer import LLM
    from microinfer.config.engine import EngineConfig

    with _nvtx_range("microinfer.load_model", nvtx):
        return LLM.from_pretrained(
            args.model,
            device=args.device,
            dtype=args.dtype,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            engine_config=EngineConfig(
                max_num_seqs=max(16, workload.num_requests),
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_model_len=4096,
            ),
            use_compile=args.use_compile,
        )


def _warmup(llm, prompts, workload, *, nvtx: bool) -> None:
    from microinfer import SampleParameter

    device = torch.device(llm.engine.model_runner.device)
    with _nvtx_range("microinfer.warmup", nvtx):
        llm.generate(
            prompts,
            SampleParameter(
                temperature=1.0,
                max_tokens=min(4, workload.output_len),
                ignore_eos=True,
                greedy=True,
            ),
            use_tqdm=False,
        )
        _sync(device)


def _measured_run(llm, prompts, params, *, nvtx: bool) -> float:
    device = torch.device(llm.engine.model_runner.device)
    with _nvtx_range("microinfer.generate", nvtx):
        started = time.perf_counter()
        llm.generate(prompts, params, use_tqdm=False)
        _sync(device)
        wall_s = time.perf_counter() - started
    return wall_s


def _reset_cache(args, llm) -> None:
    if not args.cold:
        return
    from bench.engines.common import reset_prefix_cache

    reset_prefix_cache(llm.engine.scheduler.block_manager)


def _params(workload, output_len: int):
    from microinfer import SampleParameter

    return SampleParameter(
        temperature=1.0,
        max_tokens=output_len,
        ignore_eos=True,
        greedy=True,
    )


def _run_only(args, workload, prompts, output_len: int) -> int:
    """No torch.profiler; used directly and as the nsys child."""
    device = torch.device(args.device)
    nvtx = device.type == "cuda"
    llm = _load_model(args, workload, nvtx=nvtx)

    _warmup(llm, prompts, workload, nvtx=nvtx)
    _reset_cache(args, llm)
    wall_s = _measured_run(llm, prompts, _params(workload, output_len), nvtx=nvtx)
    tokens = workload.num_requests * output_len
    print(
        f"workload={args.workload} output_tokens={tokens} wall={wall_s:.3f}s "
        f"throughput={tokens / wall_s:.1f} tok/s"
    )
    return 0


def _run_torch(args, workload, prompts, output_len: int) -> int:
    device = torch.device(args.device)
    llm = _load_model(args, workload, nvtx=False)
    _warmup(llm, prompts, workload, nvtx=False)
    _reset_cache(args, llm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.workload}_bs{args.block_size}"

    handler = None
    tb_dir = output_dir / "tensorboard"
    if args.export in ("torch", "tensorboard"):
        tb_dir.mkdir(parents=True, exist_ok=True)
        handler = torch.profiler.tensorboard_trace_handler(
            str(tb_dir), worker_name=stem, use_gzip=True
        )

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profile_kwargs = {"activities": activities, "record_shapes": False}
    if handler is not None:
        profile_kwargs["on_trace_ready"] = handler

    with torch.profiler.profile(**profile_kwargs) as prof:
        wall_s = _measured_run(
            llm, prompts, _params(workload, output_len), nvtx=False
        )

    tb_files = sorted(tb_dir.glob(f"{stem}*.pt.trace.json*")) if handler else []
    chrome_path = None
    if args.export in ("torch", "chrome"):
        chrome_path = output_dir / f"{stem}.json"
        if tb_files:
            # Kineto refuses to export the same trace twice. The TensorBoard
            # handler already saved it; reuse that JSON instead of profiling twice.
            if tb_files[-1].suffix == ".gz":
                with gzip.open(tb_files[-1], "rb") as source, open(
                    chrome_path, "wb"
                ) as destination:
                    shutil.copyfileobj(source, destination)
            else:
                shutil.copyfile(tb_files[-1], chrome_path)
        else:
            prof.export_chrome_trace(str(chrome_path))
    averages = prof.key_averages()
    output_tokens = workload.num_requests * output_len
    by_name = {
        (getattr(event, "key", None) or getattr(event, "name", "")): event
        for event in averages
    }

    def count(name: str) -> int:
        event = by_name.get(name)
        return int(event.count) if event is not None else 0

    launches = count("cudaLaunchKernel")
    syncs = count("cudaStreamSynchronize")
    gemms = count("aten::mm")
    table_cuda = averages.table(sort_by="self_cuda_time_total", row_limit=args.rows)
    table_cpu = averages.table(sort_by="self_cpu_time_total", row_limit=args.rows)

    summary = [
        "# Profiling summary",
        "",
        f"- workload: {args.workload}",
        f"- requests: {workload.num_requests}",
        f"- prompt tokens: {workload.prompt_tokens}",
        f"- output tokens: {output_tokens}",
        f"- wall (profiled): {wall_s:.3f} s",
        f"- output throughput (profiled): {output_tokens / wall_s:.1f} tok/s",
        f"- cudaLaunchKernel: {launches} ({launches / output_tokens:.1f}/token)",
        f"- cudaStreamSynchronize: {syncs} ({syncs / output_tokens:.1f}/token)",
        f"- aten::mm: {gemms} ({gemms / output_tokens:.1f}/token)",
        "- note: profiling instruments every launch; use these ratios for hotspot "
        "attribution, not absolute throughput",
        "",
        "## Top CUDA activities",
        "",
        "```",
        table_cuda,
        "```",
        "",
        "## Top CPU activities",
        "",
        "```",
        table_cpu,
        "```",
    ]
    if chrome_path is not None:
        summary.append(f"Chrome/Perfetto trace: {chrome_path}")
    if tb_files:
        summary.append(f"TensorBoard trace: {tb_files[-1]}")
        summary.append(f"TensorBoard logdir: {tb_dir}")
    summary_path = output_dir / f"{stem}_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n")

    print("\n".join(summary[:15]))
    print()
    if chrome_path is not None:
        print(f"chrome/perfetto trace: {chrome_path}")
        print("  open with chrome://tracing or https://ui.perfetto.dev")
    if tb_files:
        print(f"tensorboard trace: {tb_files[-1]}")
        print(f"  view with: tensorboard --logdir {tb_dir}")
        if importlib.util.find_spec("tensorboard") is None:
            print("  note: tensorboard is not installed; install it to view this trace")
    print(f"summary: {summary_path}")
    return 0


def _run_nsys(args, workload) -> int:
    nsys_bin = args.nsys_bin or shutil.which("nsys")
    if not nsys_bin:
        print(
            "nsys not found. Install Nsight Systems or pass --nsys-bin.\n"
            "Fallback: --export torch (Chrome/Perfetto + TensorBoard).",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.workload}_bs{args.block_size}"
    output_base = output_dir / stem

    child_argv = [
        "-m",
        "bench.profile_runner",
        "--model",
        args.model,
        "--workload",
        args.workload,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--block-size",
        str(args.block_size),
        "--num-blocks",
        str(args.num_blocks),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
        "--export",
        "run",
    ]
    if args.max_tokens is not None:
        child_argv += ["--max-tokens", str(args.max_tokens)]
    if args.cold:
        child_argv.append("--cold")
    if args.use_compile:
        child_argv.append("--use-compile")

    command = nsys_command(
        nsys_bin,
        output_base,
        args.nsys_trace,
        child_argv,
        python_exe=sys.executable,
    )
    report = output_base.with_suffix(".nsys-rep")
    print("command:", " ".join(shlex.quote(part) for part in command))
    print(f"nsys report will be: {report}")
    print(f"  open with: nsys-ui {report}")
    print(f"  or: nsys stats --report cuda_gpu_kern_sum {report}")
    sys.stdout.flush()

    # Replace this Python process instead of spawning nsys from a Python parent.
    # A torch-loaded parent can leave the nsys agent waiting during shutdown.
    try:
        os.chdir(Path(__file__).resolve().parents[1])
        os.execv(nsys_bin, command)
    except OSError as exc:
        print(f"failed to launch nsys: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", choices=WORKLOAD_NAMES, default="single-short")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override workload output length (smaller traces)",
    )
    parser.add_argument("--rows", type=int, default=25)
    parser.add_argument(
        "--cold",
        action="store_true",
        help="drop prefix-cache entries before the profiled run",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="compile the decode forward before profiling",
    )
    parser.add_argument(
        "--export",
        choices=EXPORT_CHOICES,
        default="torch",
        help="torch: Chrome + TensorBoard; nsys: .nsys-rep; run: no profiler",
    )
    parser.add_argument(
        "--nsys-bin", default=None, help="path to the nsys executable"
    )
    parser.add_argument(
        "--nsys-trace",
        default="cuda,nvtx",
        help="Nsight Systems --trace value",
    )
    parser.add_argument("--output-dir", default="results/bench/traces")
    args = parser.parse_args()

    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    workload = build_workload(args.workload, seed=args.seed)
    prompts = [list(prompt) for prompt in workload.prompts]
    output_len = args.max_tokens or workload.output_len

    if args.export == "nsys":
        return _run_nsys(args, workload)
    if args.export == "run":
        return _run_only(args, workload, prompts, output_len)
    return _run_torch(args, workload, prompts, output_len)


if __name__ == "__main__":
    raise SystemExit(main())
