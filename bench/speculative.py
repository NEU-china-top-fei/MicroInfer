"""Compare greedy generation with and without n-gram speculative decoding."""

from __future__ import annotations

import argparse
import json
import time

from microinfer import LLM, SampleParameter
from microinfer.config.engine import EngineConfig

DEFAULT_PROMPT = (
    "Repeat exactly the following sentence, then continue repeating it: "
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog."
)


def _run(model, prompt, params, *, speculative, block_size, num_blocks, graph, runs):
    config = EngineConfig(
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        max_num_prefill_tokens=2048,
        max_model_len=4096,
        use_speculative=speculative,
    )
    llm = LLM.from_pretrained(
        model,
        block_size=block_size,
        num_blocks=num_blocks,
        engine_config=config,
        use_cuda_graph=graph,
        use_speculative=speculative,
    )
    llm.generate([prompt], params, use_tqdm=False)
    manager = llm.engine.scheduler.block_manager
    manager.reset_prefix_cache()
    times = []
    token_ids = None
    for _ in range(runs):
        manager.reset_prefix_cache()
        start = time.perf_counter()
        output = llm.generate([prompt], params, use_tqdm=False)[0]
        times.append(time.perf_counter() - start)
        token_ids = output["token_ids"]
    stats = llm.engine.stats()
    return {
        "token_ids": token_ids,
        "times_s": times,
        "median_s": sorted(times)[len(times) // 2],
        "output_tps": len(token_ids) / sorted(times)[len(times) // 2],
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--use-cuda-graph", action="store_true")
    args = parser.parse_args()
    params = SampleParameter(
        temperature=1.0,
        max_tokens=args.max_tokens,
        greedy=True,
        ignore_eos=True,
    )
    baseline = _run(
        args.model,
        args.prompt,
        params,
        speculative=False,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        graph=args.use_cuda_graph,
        runs=args.runs,
    )
    speculative = _run(
        args.model,
        args.prompt,
        params,
        speculative=True,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        graph=args.use_cuda_graph,
        runs=args.runs,
    )
    summary = {
        "prompt_chars": len(args.prompt),
        "max_tokens": args.max_tokens,
        "baseline": baseline,
        "speculative": speculative,
        "outputs_match": baseline["token_ids"] == speculative["token_ids"],
        "speedup": baseline["median_s"] / speculative["median_s"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
