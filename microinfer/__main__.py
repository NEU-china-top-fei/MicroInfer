import argparse
import json

from microinfer import LLM, SampleParameter


def main():
    parser = argparse.ArgumentParser(
        description="Generate with a local Qwen3 checkpoint"
    )
    parser.add_argument(
        "--model", required=True, help="local safetensors/tokenizer directory"
    )
    parser.add_argument(
        "--prompt", action="append", required=True, help="repeat for a batch"
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=64)
    args = parser.parse_args()
    llm = LLM.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
    )
    results = llm.generate(
        args.prompt,
        SampleParameter(args.temperature, args.max_tokens, greedy=args.greedy),
        use_tqdm=False,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
