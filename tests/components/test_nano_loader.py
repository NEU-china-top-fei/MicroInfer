import pytest
import torch
from torch import nn

from microinfer.model.loader import load_model

pytestmark = [pytest.mark.component, pytest.mark.nano]


def test_load_local_safetensors_without_replacing_parameter(tmp_path):
    from safetensors.torch import save_file

    model = nn.Linear(3, 2, bias=False)
    identity = id(model.weight)
    expected = torch.arange(6.0).reshape(2, 3)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    load_model(model, str(tmp_path))
    assert id(model.weight) == identity
    torch.testing.assert_close(model.weight, expected)
