"""Launch the OpenAI-compatible server: python -m microinfer.serve --model DIR."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Serve a local Qwen3 checkpoint")
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--served-model-name", default="microinfer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-prefill-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-waiting", type=int, default=128)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--no-fuse-weights", action="store_true")
    parser.add_argument("--no-mixed-batch", action="store_true")
    parser.add_argument("--compile-ops", action="store_true")
    parser.add_argument("--use-speculative", action="store_true")
    parser.add_argument("--draft-model", default=None)
    parser.add_argument(
        "--draft-layers", type=int, default=None, help="use a layer-pruned native draft"
    )
    parser.add_argument("--draft-num-blocks", type=int, default=None)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "serve dependencies are missing; install with: pip install -e '.[hf,serve]'"
        ) from exc

    from microinfer import LLM
    from microinfer.config.engine import EngineConfig
    from microinfer.serve.app import create_app

    model_path = Path(args.model)
    max_model_len = args.max_model_len
    if max_model_len is None:
        import json

        settings = json.loads((model_path / "config.json").read_text())
        max_model_len = min(
            int(settings.get("max_position_embeddings", 4096)),
            args.block_size * args.num_blocks,
        )
    engine_config = EngineConfig(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_prefill_tokens=args.max_num_prefill_tokens,
        max_model_len=max_model_len,
        mixed_batch=not args.no_mixed_batch,
        use_speculative=args.use_speculative,
    )
    llm = LLM.from_pretrained(
        model_path,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        engine_config=engine_config,
        use_compile=args.use_compile,
        use_cuda_graph=args.use_cuda_graph,
        compile_ops=args.compile_ops or None,
        use_speculative=args.use_speculative or args.draft_model is not None,
        draft_model_path=args.draft_model,
        draft_layers=args.draft_layers,
        draft_num_blocks=args.draft_num_blocks,
        fuse_weights=not args.no_fuse_weights,
    )
    app = create_app(
        llm,
        model_name=args.served_model_name,
        max_waiting=args.max_waiting,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
