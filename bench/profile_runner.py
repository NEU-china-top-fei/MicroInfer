"""Capture a torch.profiler trace for one MicroInfer workload."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from bench.workloads import WORKLOAD_NAMES, build_workload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", choices=WORKLOAD_NAMES, default="single-short")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
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
    parser.add_argument("--output-dir", default="results/bench/traces")
    args = parser.parse_args()

    from microinfer import LLM, SampleParameter
    from microinfer.config.engine import EngineConfig

    workload = build_workload(args.workload, seed=args.seed)
    llm = LLM.from_pretrained(
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
    )
    prompts = [list(prompt) for prompt in workload.prompts]
    output_len = args.max_tokens or workload.output_len
    params = SampleParameter(
        temperature=1.0,
        max_tokens=output_len,
        ignore_eos=True,
        greedy=True,
    )

    # Compile/autotune, module loading, and allocator growth happen here.
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
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()

    if args.cold:
        from bench.engines.common import reset_prefix_cache

        reset_prefix_cache(llm.engine.scheduler.block_manager)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(args.device).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        started = time.perf_counter()
        llm.generate(prompts, params, use_tqdm=False)
        wall_s = time.perf_counter() - started
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.workload}_bs{args.block_size}"
    trace_path = output_dir / f"{stem}.json"
    summary_path = output_dir / f"{stem}_summary.txt"
    prof.export_chrome_trace(str(trace_path))

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
        "",
        f"Chrome trace: {trace_path}",
    ]
    summary_path.write_text("\n".join(summary) + "\n")
    print("\n".join(summary[:15]))
    print(f"\ntrace: {trace_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
