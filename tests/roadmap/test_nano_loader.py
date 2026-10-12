from typing import ClassVar

import pytest
import torch
from torch import nn

from microinfer.model.loader import load_model

pytestmark = [pytest.mark.roadmap, pytest.mark.nano]


def test_load_local_safetensors_without_replacing_parameter(tmp_path):
    from safetensors.torch import save_file

    model = nn.Linear(3, 2, bias=False)
    identity = id(model.weight)
    expected = torch.arange(6.0).reshape(2, 3)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    load_model(model, str(tmp_path))
    assert id(model.weight) == identity
    torch.testing.assert_close(model.weight, expected)


def test_packed_weights_dispatch_named_shards(tmp_path):
    from safetensors.torch import save_file

    class PackedModel(nn.Module):
        packed_modules_mapping: ClassVar = {
            f"{name}_proj": ("qkv_proj", name) for name in ["q", "k", "v"]
        }

        def __init__(self):
            super().__init__()
            self.qkv_proj = nn.Linear(2, 6, bias=False)
            self.loaded = []

            def load(param, tensor, shard_id):
                index = {"q": 0, "k": 1, "v": 2}[shard_id]
                param.data[index * 2 : (index + 1) * 2].copy_(tensor)
                self.loaded.append(shard_id)

            self.qkv_proj.weight.weight_loader = load

    model = PackedModel()
    weights = {
        f"{name}_proj.weight": torch.full((2, 2), float(i))
        for i, name in enumerate(["q", "k", "v"], 1)
    }
    save_file(weights, tmp_path / "model.safetensors")
    load_model(model, str(tmp_path))
    assert sorted(model.loaded) == ["k", "q", "v"]
    torch.testing.assert_close(model.qkv_proj.weight, torch.cat(list(weights.values())))
