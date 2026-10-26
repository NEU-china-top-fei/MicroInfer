"""Concurrent online benchmark over the async engine (no HTTP overhead)."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from bench.metrics import describe
from microinfer import SampleParameter
from microinfer.config.engine import EngineConfig
from microinfer.engine.async_engine import AsyncEngine
from microinfer.llm import LLM


def _prompts(count, length, seed):
    rng = random.Random(seed)
    return [
        tuple(rng.randrange(100, 20_000) for _ in range(length)) for _ in range(count)
    ]


async def _run(llm, prompts, output_len, late_prompts=(), late_after_ms=0):
    async_engine = AsyncEngine(llm, max_waiting=len(prompts) + len(late_prompts) + 1)
    params = SampleParameter(
        temperature=1.0,
        max_tokens=output_len,
        greedy=True,
        ignore_eos=True,
    )
    submitted = time.perf_counter()

    async def one(prompt, wave):
        handle = await async_engine.add_request(list(prompt), params)
        token_times = []
        async for event in async_engine.stream(handle):
            if event.token_ids:
                token_times.append(time.perf_counter())
        return {
            "wave": wave,
            "ttft_s": token_times[0] - submitted if token_times else 0.0,
            "token_times_s": token_times,
            "output_tokens": len(handle.token_ids),
        }

    tasks = [asyncio.create_task(one(prompt, 0)) for prompt in prompts]
    if late_prompts:
        await asyncio.sleep(late_after_ms / 1000)
        tasks.extend(asyncio.create_task(one(prompt, 1)) for prompt in late_prompts)
    results = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - submitted
    await async_engine.stop()
    return results, wall_s, async_engine.stats()


def _wave_summary(results):
    ttfts = [r["ttft_s"] for r in results]
    tpots = [
        (times[-1] - times[0]) / max(1, len(times) - 1)
        for r in results
        for times in [r["token_times_s"]]
        if len(times) > 1
    ]
    itls = [
        (later - earlier) * 1000
        for r in results
        for earlier, later in zip(r["token_times_s"], r["token_times_s"][1:])
    ]
    return {
        "requests": len(results),
        "output_tokens": sum(r["output_tokens"] for r in results),
        "ttft_ms": describe([x * 1000 for x in ttfts]),
        "tpot_ms": describe([x * 1000 for x in tpots]),
        "itl_ms": describe(itls),
    }


def _summary(results, wall_s, stats, output_len):
    output_tokens = sum(r["output_tokens"] for r in results)
    waves = {
        str(wave): _wave_summary([r for r in results if r["wave"] == wave])
        for wave in sorted({r["wave"] for r in results})
    }
    return {
        "requests": len(results),
        "output_tokens": output_tokens,
        "wall_s": wall_s,
        "output_tps": output_tokens / wall_s if wall_s else 0.0,
        "ttft_ms": describe([r["ttft_s"] * 1000 for r in results]),
        "tpot_ms": describe(
            [
                (times[-1] - times[0]) / max(1, len(times) - 1) * 1000
                for r in results
                for times in [r["token_times_s"]]
                if len(times) > 1
            ]
        ),
        "by_wave": waves,
        "engine_stats": stats,
        "requested_output_len": output_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--late-requests", type=int, default=0)
    parser.add_argument("--late-prompt-len", type=int, default=1024)
    parser.add_argument("--late-after-ms", type=int, default=100)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-prefill-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--no-fuse-weights", action="store_true")
    parser.add_argument(
        "--no-mixed-batch",
        action="store_true",
        help="use the pre-P4 prefill-first scheduler for A/B",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = EngineConfig(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_prefill_tokens=args.max_num_prefill_tokens,
        max_model_len=min(4096, args.block_size * args.num_blocks),
        mixed_batch=not args.no_mixed_batch,
    )
    llm = LLM.from_pretrained(
        args.model,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        engine_config=config,
        use_compile=args.use_compile,
        use_cuda_graph=args.use_cuda_graph,
        fuse_weights=not args.no_fuse_weights,
    )
    warmup = SampleParameter(
        temperature=1.0, max_tokens=2, greedy=True, ignore_eos=True
    )
    llm.generate([[1, 2, 3, 4]], warmup, use_tqdm=False)

    prompts = _prompts(args.requests, args.prompt_len, args.seed)
    late_prompts = _prompts(args.late_requests, args.late_prompt_len, args.seed + 1)
    results, wall_s, stats = asyncio.run(
        _run(
            llm,
            prompts,
            args.output_len,
            late_prompts=late_prompts,
            late_after_ms=args.late_after_ms,
        )
    )
    summary = _summary(results, wall_s, stats, args.output_len)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
