"""Concurrent, text-prompt benchmark for draft-model speculative decoding."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from bench.metrics import describe
from microinfer import LLM, AsyncEngine, SampleParameter
from microinfer.config.engine import EngineConfig

PROMPTS = [
    (
        "Write a Python function that merges two sorted lists and explain its "
        "time complexity."
    ),
    (
        "Return a JSON object describing a user with name, age, and a list of "
        "three hobbies."
    ),
    (
        "Summarize in two sentences: PagedAttention stores KV cache in fixed-size "
        "blocks and uses a block table to map logical tokens to physical blocks, "
        "which reduces fragmentation."
    ),
    (
        "A customer reports that an API sometimes returns stale data after a "
        "write. Describe the likely causes and a debugging plan."
    ),
    (
        "Repeat exactly: alpha beta gamma delta. alpha beta gamma delta. "
        "alpha beta gamma delta."
    ),
    (
        "Explain the difference between latency and throughput in an inference "
        "service, with one concrete example."
    ),
]


async def _run(llm, prompts, params, max_waiting):
    engine = AsyncEngine(llm, max_waiting=max_waiting)
    submitted = time.perf_counter()

    async def one(prompt):
        handle = await engine.add_request(prompt, params)
        token_times = []
        async for event in engine.stream(handle):
            if event.token_ids:
                token_times.append(time.perf_counter())
        return {
            "ttft_s": token_times[0] - submitted if token_times else 0.0,
            "token_times_s": token_times,
            "token_ids": handle.token_ids,
        }

    results = await asyncio.gather(*(one(prompt) for prompt in prompts))
    wall_s = time.perf_counter() - submitted
    stats = engine.stats()
    await engine.stop()
    return results, wall_s, stats


def _summary(results, wall_s, stats):
    ttfts = [r["ttft_s"] for r in results]
    tpots = [
        (times[-1] - times[0]) / max(1, len(times) - 1)
        for r in results
        for times in [r["token_times_s"]]
        if len(times) > 1
    ]
    completion_tokens = sum(len(r["token_ids"]) for r in results)
    return {
        "requests": len(results),
        "completion_tokens": completion_tokens,
        "wall_s": wall_s,
        "output_tps": completion_tokens / wall_s if wall_s else 0.0,
        "ttft_ms": describe([x * 1000 for x in ttfts]),
        "tpot_ms": describe([x * 1000 for x in tpots]),
        "spec": {
            key: stats.get(key)
            for key in (
                "num_spec_steps",
                "num_spec_draft_tokens",
                "num_spec_accepted_tokens",
                "spec_acceptance_rate",
            )
        },
    }


def _load(model, draft_model, args):
    config = EngineConfig(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_prefill_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        use_speculative=draft_model is not None,
        speculative_max_drafts=args.max_drafts,
    )
    return LLM.from_pretrained(
        model,
        device="cuda",
        dtype="bfloat16",
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        engine_config=config,
        use_cuda_graph=args.use_cuda_graph,
        compile_ops=args.compile_ops or None,
        use_speculative=draft_model is not None,
        draft_model_path=draft_model,
        draft_layers=args.draft_layers,
        draft_num_blocks=args.draft_num_blocks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-layers", type=int, default=None)
    parser.add_argument("--draft-num-blocks", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-drafts", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--compile-ops", action="store_true")
    parser.add_argument("--prompts", choices=("builtin",), default="builtin")
    args = parser.parse_args()

    params = SampleParameter(
        temperature=1.0,
        max_tokens=args.output_len,
        greedy=True,
        ignore_eos=True,
    )
    prompts = PROMPTS

    def run(mode, draft_model):
        llm = _load(args.model, draft_model, args)
        llm.generate(
            ["Warm up the model and CUDA graph."],
            SampleParameter(max_tokens=2, greedy=True, ignore_eos=True),
            use_tqdm=False,
        )
        llm.engine.scheduler.block_manager.reset_prefix_cache()
        all_runs = []
        for _ in range(args.runs):
            llm.engine.scheduler.block_manager.reset_prefix_cache()
            results, wall_s, stats = asyncio.run(
                _run(llm, prompts, params, args.max_num_seqs * 2)
            )
            all_runs.append(_summary(results, wall_s, stats))
        best = min(all_runs, key=lambda item: item["wall_s"])
        return {
            "mode": mode,
            "tokens": [list(r["token_ids"]) for r in results],
            "best": best,
            "runs": all_runs,
        }

    baseline = run("baseline", None)
    speculative = run("draft_model", args.draft_model)
    match = baseline["tokens"] == speculative["tokens"]
    output = {
        "model": args.model,
        "draft_model": args.draft_model,
        "prompts": prompts,
        "output_len": args.output_len,
        "baseline": baseline,
        "speculative": speculative,
        "outputs_match": match,
        "speculative_speedup": speculative["best"]["output_tps"]
        / baseline["best"]["output_tps"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
