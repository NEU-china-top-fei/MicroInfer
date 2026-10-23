"""Local safetensors loading; model adapters map checkpoint names explicitly."""

from pathlib import Path

import torch


@torch.no_grad()
def default_weight_loader(param, loaded_weight):
    if param.shape != loaded_weight.shape:
        raise ValueError(
            f"weight shape mismatch: {tuple(loaded_weight.shape)} != {tuple(param.shape)}"
        )
    param.copy_(loaded_weight)


def qwen_name(name):
    replacements = (
        ("model.embed_tokens.", "embedding."),
        ("model.layers.", "transformer_layers."),
        ("model.norm.", "mid_norm.norm."),
        ("lm_head.", "out_logits."),
        (".self_attn.", ".attention."),
        (".o_proj.", ".out_proj."),
        (".q_norm.", ".q_norm.norm."),
        (".k_norm.", ".k_norm.norm."),
        (".input_layernorm.", ".pre_attention_norm.norm."),
        (".post_attention_layernorm.", ".post_attention_norm.norm."),
        (".mlp.gate_proj.", ".ffn.gate_project."),
        (".mlp.up_proj.", ".ffn.up_project."),
        (".mlp.down_proj.", ".ffn.down_project."),
    )
    for source, target in replacements:
        name = name.replace(source, target)
    return name


def load_model(model, path):
    from safetensors import safe_open

    from microinfer.model.qwen3 import QwenModel

    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint in {path}")
    params = dict(model.named_parameters(remove_duplicate=False))
    seen, loaded = set(), set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for source in checkpoint.keys():  # noqa: SIM118 -- safe_open is not iterable
                if source in seen:
                    raise ValueError(f"duplicate checkpoint key: {source}")
                seen.add(source)
                name = qwen_name(source) if isinstance(model, QwenModel) else source
                if name not in params:
                    raise ValueError(f"unexpected checkpoint weight: {source}")
                param = params[name]
                default_weight_loader(param, checkpoint.get_tensor(source))
                loaded.add(id(param))
    missing = [name for name, param in params.items() if id(param) not in loaded]
    if missing:
        raise ValueError(f"missing checkpoint weights: {missing}")
