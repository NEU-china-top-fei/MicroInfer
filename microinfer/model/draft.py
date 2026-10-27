"""Load a smaller layer-pruned Qwen draft from a full Qwen3 checkpoint."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open

from microinfer.config import QwenConfig
from microinfer.model.loader import default_weight_loader, qwen_name
from microinfer.model.qwen3 import QwenModel


@torch.no_grad()
def load_pruned_draft(path, *, num_layers: int, dtype, device):
    path = Path(path)
    settings = json.loads((path / "config.json").read_text())
    config = QwenConfig.from_hf(settings)
    if type(num_layers) is not int or not 1 <= num_layers <= config.num_hidden_layer:
        raise ValueError("draft layers must be between 1 and the checkpoint layers")
    config = replace(config, num_hidden_layer=num_layers)

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = QwenModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    params = dict(model.named_parameters(remove_duplicate=False))
    loaded = set()
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for source in checkpoint.keys():  # noqa: SIM118
                if source.startswith("model.layers."):
                    layer = int(source.split(".")[2])
                    if layer >= num_layers:
                        continue
                target = qwen_name(source)
                if target not in params:
                    raise ValueError(f"unexpected draft checkpoint weight: {source}")
                param = params[target]
                default_weight_loader(param, checkpoint.get_tensor(source))
                loaded.add(id(param))
    missing = [name for name, param in params.items() if id(param) not in loaded]
    if missing:
        raise ValueError(f"missing draft checkpoint weights: {missing}")
    model.eval()
    model.fuse_weights()
    return model
