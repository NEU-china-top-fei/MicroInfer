import pytest
import torch
from torch.nn import functional as F

from microinfer.layer.transformer import FFN


def test_ffn_projection_dimensions(tiny_config):
    layer = FFN(tiny_config)
    assert layer.up_project.weight.shape == (24, 16)
    assert layer.gate_project.weight.shape == (24, 16)
    assert layer.down_project.weight.shape == (16, 24)


@pytest.mark.known_bug(
    reason="Qwen3 MLP projections must be bias-free for checkpoint compatibility"
)
def test_qwen_ffn_has_no_projection_bias(tiny_config):
    layer = FFN(tiny_config)
    assert all(
        proj.bias is None
        for proj in (layer.up_project, layer.gate_project, layer.down_project)
    )


@pytest.mark.known_bug(reason="FFN defines foward; SwiGLU must activate gate_project")
@pytest.mark.parametrize("shape", [(3, 16), (2, 3, 16)])
def test_ffn_matches_swiglu_formula(tiny_config, seeded_torch, device, shape):
    layer = FFN(tiny_config).to(device)
    x = torch.randn(*shape).to(device)
    up = F.linear(x, layer.up_project.weight, layer.up_project.bias)
    gate = F.linear(x, layer.gate_project.weight, layer.gate_project.bias)
    expected = F.linear(
        F.silu(gate) * up, layer.down_project.weight, layer.down_project.bias
    )
    actual = layer(x)  # Public nn.Module dispatch, not the misspelled method.
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)
