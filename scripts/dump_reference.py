import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(args.device)
        .eval()
    )
    # 先用原始文本；chat template 留到数值链路正确以后。
    ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(args.device)
    with torch.inference_mode():
        output = model(ids, use_cache=False, output_hidden_states=True)
        embedding = model.get_input_embeddings()(ids)
    out_dir = Path("results/reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": ids.cpu(),
            "embedding": embedding.cpu(),
            "hidden_states": tuple(x.cpu() for x in output.hidden_states),
            "logits": output.logits.cpu(),
        },
        out_dir / "case01.pt",
    )
    meta = {
        "model_path": str(Path(args.model).resolve()),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "dtype": "float32",
        "attention": "eager",
        "config": model.config.to_dict(),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("input_ids:", ids.cpu().tolist())
    print("logits shape:", tuple(output.logits.shape))
    print("next_token_id:", output.logits[0, -1].argmax().item())


if __name__ == "__main__":
    main()
