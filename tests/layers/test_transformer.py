import torch
from torch import nn

from microinfer.layer.transformer import TransformerLayer


class PositionAttention(nn.Module):
    def forward(self, x, position):
        assert position.shape == (x.shape[1],)
        return x * 0.25 + position[None, :, None]


class DoubleFFN(nn.Module):
    def forward(self, x):
        return x * 2


def test_transformer_wires_positions_and_residuals(tiny_config, seeded_torch):
    # Isolate residual orchestration from the attention and FFN kernels.
    tiny_config.rope_base = 10000.0
    layer = TransformerLayer(tiny_config)
    layer.attention, layer.ffn = PositionAttention(), DoubleFFN()
    x = torch.randn(2, 3, tiny_config.hidden_dim)
    residual = torch.randn_like(x)
    positions = torch.arange(3)

    def rms(value):
        return value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + tiny_config.norm_eps
        )

    combined = x + residual
    attention_output = rms(combined) * 0.25 + positions[None, :, None]
    expected_residual = combined + attention_output
    actual, carried = layer(x, positions, residual)
    torch.testing.assert_close(carried, expected_residual)
    torch.testing.assert_close(actual, 2 * rms(expected_residual))
