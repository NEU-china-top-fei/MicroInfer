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
    packed = getattr(model, "packed_modules_mapping", {})
    seen, loaded = set(), set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for source in checkpoint.keys():  # noqa: SIM118 -- safe_open is not iterable
                if source in seen:
                    raise ValueError(f"duplicate checkpoint key: {source}")
                seen.add(source)
                name = qwen_name(source) if isinstance(model, QwenModel) else source
                shard = None
                for source_module, (target_module, shard_id) in packed.items():
                    if source_module in name.split("."):
                        name = ".".join(
                            target_module if part == source_module else part
                            for part in name.split(".")
                        )
                        shard = shard_id
                        break
                if name not in params:
                    raise ValueError(f"unexpected checkpoint weight: {source}")
                param = params[name]
                tensor = checkpoint.get_tensor(source)
                loader = getattr(param, "weight_loader", default_weight_loader)
                if shard is None:
                    loader(param, tensor)
                else:
                    loader(param, tensor, shard)
                loaded.add(id(param))
    missing = [name for name, param in params.items() if id(param) not in loaded]
    if missing:
        raise ValueError(f"missing checkpoint weights: {missing}")


@torch.no_grad()
def load_weights(model, weights, name_map=None):
    targets = model.state_dict(keep_vars=True)
    mapping, converted = name_map or {}, {}
    for source, value in weights.items():
        target = mapping.get(source, source)
        if (
            target in converted
            or target not in targets
            or value.shape != targets[target].shape
        ):
            raise ValueError(
                f"invalid checkpoint destination/shape: {source} -> {target}"
            )
        converted[target] = value.to(
            device=targets[target].device, dtype=targets[target].dtype
        )
    if set(converted) != set(targets):
        raise ValueError("checkpoint keys do not match model")
    for name, value in converted.items():
        targets[name].copy_(value)
