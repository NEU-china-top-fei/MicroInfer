import argparse
import json

from microinfer import LLM, SampleParameter


def main():
    parser = argparse.ArgumentParser(
        description="Generate with a local Qwen3 checkpoint"
    )
    parser.add_argument(
        "--debug-mode",
        "--debug_mode",
        action="store_true",
        dest="debug_mode",
        help="enable scheduler/runner invariant checks",
    )
    parser.add_argument(
        "--use-compile",
        "--use_compile",
        action="store_true",
        dest="use_compile",
        help="compile the decode forward (mutually exclusive with CUDA Graph)",
    )
    parser.add_argument(
        "--no-fuse-weights",
        action="store_false",
        dest="fuse_weights",
        help="keep separate QKV/gate-up weights (lower memory, more GEMMs)",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="capture decode buckets with CUDA Graph (mutually exclusive with compile)",
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
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV block size; powers of two use the Triton paged path below 256",
    )
    parser.add_argument(
        "--num-blocks", type=int, default=64, help="number of physical KV blocks"
    )
    args = parser.parse_args()
    llm = LLM.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        debug_mode=args.debug_mode,
        use_compile=args.use_compile,
        fuse_weights=args.fuse_weights,
        use_cuda_graph=args.use_cuda_graph,
    )
    results = llm.generate(
        args.prompt,
        SampleParameter(args.temperature, args.max_tokens, greedy=args.greedy),
        use_tqdm=False,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
