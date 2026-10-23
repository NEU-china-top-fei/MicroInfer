"""Compare a local dense Qwen3 checkpoint with Transformers, sequentially in RAM."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from microinfer import LLM, SampleParameter
    from microinfer.engine.sequence import Sequence

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(args.device)
    print("Loading Transformers reference", flush=True)
    oracle = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(args.device)
        .eval()
    )
    with torch.inference_mode():
        expected = oracle(ids, use_cache=False).logits.cpu()
        generated = ids.clone()
        for _ in range(3):
            next_token = (
                oracle(generated, use_cache=False)
                .logits[:, -1]
                .argmax(-1, keepdim=True)
            )
            generated = torch.cat((generated, next_token), dim=1)
    del oracle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Loading MicroInfer", flush=True)
    llm = LLM.from_pretrained(
        args.model, device=args.device, dtype=torch.float32, block_size=4, num_blocks=8
    )
    runner = llm.engine.model_runner
    with torch.inference_mode():
        actual = runner.model.compute_logits(
            runner.model(ids, torch.arange(ids.shape[1], device=args.device))
        ).cpu()
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        # Exercise an actual cached prefix, ending with decode, against the same oracle.
        seq = Sequence(ids[0].tolist())
        seq.block_size = 4
        seq.block_table = list(range((len(seq) + 3) // 4))
        chunk_errors = []
        start = 0
        for end in [2, len(seq) - 1, len(seq)]:
            if end <= start:
                continue
            seq.num_cached_tokens, seq.num_scheduled_tokens = start, end - start
            prefill = end != len(seq)
            if prefill:
                ids_chunk, positions, metadata = runner.prepare_prefill([seq])
            else:
                ids_chunk, positions, metadata = runner.prepare_decode([seq])
            logits = runner.run_model(ids_chunk, positions, prefill, metadata)[0].cpu()
            torch.testing.assert_close(
                logits, expected[0, end - 1], atol=2e-4, rtol=2e-4
            )
            chunk_errors.append(float((logits - expected[0, end - 1]).abs().max()))
            start = end
        outputs = llm.generate(
            [ids[0].tolist()],
            SampleParameter(max_tokens=3, greedy=True, ignore_eos=True),
            use_tqdm=False,
        )
    expected_tokens = generated[0, ids.shape[1] :].tolist()
    assert outputs[0]["token_ids"] == expected_tokens
    report = {
        "model": str(Path(args.model).resolve()),
        "device": args.device,
        "dtype": "float32",
        "dense_max_abs_error": float((actual - expected).abs().max()),
        "cached_max_abs_errors": chunk_errors,
        "greedy_token_ids": expected_tokens,
        "text": outputs[0]["text"],
        "passed": True,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
