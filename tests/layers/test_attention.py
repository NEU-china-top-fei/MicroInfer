import importlib.util

import pytest
import torch
from torch.nn import functional as F

from microinfer.layer.transformer import Attention


@pytest.fixture
def attention_config(tiny_config):
    # Supply the missing config field so layout tests can isolate Attention.
    # The production default contract is separately tested in test_config.py.
    tiny_config.rope_base = 10000.0
    return tiny_config


def test_qkv_projection_dimensions(attention_config):
    layer = Attention(attention_config)
    assert layer.q_proj.weight.shape == (16, 16)
    assert layer.k_proj.weight.shape == (8, 16)
    assert layer.v_proj.weight.shape == (8, 16)


@pytest.mark.known_bug(reason="Output projection consumes query heads, not KV heads")
def test_gqa_output_projection_uses_query_width(attention_config):
    layer = Attention(attention_config)
    assert (
        layer.out_proj.in_features
        == attention_config.num_q_head * attention_config.head_dim
    )


@pytest.mark.known_bug(reason="Q/K RMSNorm must normalize head_dim, not hidden_dim")
def test_qk_norm_operates_per_head(attention_config):
    layer = Attention(attention_config)
    q = torch.randn(2, 3, attention_config.num_q_head, attention_config.head_dim)
    k = torch.randn(2, 3, attention_config.num_kv_head, attention_config.head_dim)
    for norm, x in [(layer.q_norm, q), (layer.k_norm, k)]:
        actual, _ = norm(x)
        expected = x * torch.rsqrt(
            x.square().mean(-1, keepdim=True) + attention_config.norm_eps
        )
        torch.testing.assert_close(actual, expected)


def attention_reference(layer, x, positions, config):
    """Independent FP32 causal GQA with the current adjacent-pair RoPE layout."""
    batch, length, _ = x.shape

    def project(proj, heads):
        return F.linear(
            x.float(),
            proj.weight.float(),
            None if proj.bias is None else proj.bias.float(),
        ).reshape(batch, length, heads, config.head_dim)

    def norm_rotate(tensor):
        # GPU test explicitly sets both RMS weights to one.
        tensor = tensor * torch.rsqrt(
            tensor.square().mean(-1, keepdim=True) + config.norm_eps
        )
        angles = positions.float()[:, None] / config.rope_base ** (
            torch.arange(0, config.head_dim, 2, device=x.device).float()
            / config.head_dim
        )
        cosine, sine = angles.cos()[None, :, None], angles.sin()[None, :, None]
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), -1
        ).flatten(-2)

    q = norm_rotate(project(layer.q_proj, config.num_q_head)).transpose(1, 2)
    k = norm_rotate(project(layer.k_proj, config.num_kv_head))
    v = project(layer.v_proj, config.num_kv_head)
    repeats = config.num_q_head // config.num_kv_head
    k = k.repeat_interleave(repeats, dim=2).transpose(1, 2)
    v = v.repeat_interleave(repeats, dim=2).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / config.head_dim**0.5
    mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
    probabilities = scores.masked_fill(mask, float("-inf")).softmax(-1)
    values = (probabilities @ v).transpose(1, 2).reshape(batch, length, -1)
    return F.linear(
        values,
        layer.out_proj.weight.float(),
        None if layer.out_proj.bias is None else layer.out_proj.bias.float(),
    )


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.known_bug(
    reason="Attention forward lacks working RoPE, QK normalization and varlen arguments"
)
def test_attention_matches_causal_gqa_reference(attention_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    if importlib.util.find_spec("flash_attn") is None:
        pytest.skip("FlashAttention is not installed")
    config = attention_config
    config.head_dim, config.hidden_dim = 32, 128
    layer = Attention(config).to(device="cuda", dtype=torch.float16).eval()
    with torch.no_grad():
        for norm in (layer.q_norm, layer.k_norm):
            norm.norm.weight.fill_(1)
        x = (
            torch.randn(
                2, 4, config.hidden_dim, generator=torch.Generator().manual_seed(0)
            )
            .cuda()
            .half()
        )
        positions = torch.arange(3, 7, device="cuda")
        actual = layer(x, positions)
        expected = attention_reference(layer, x, positions, config)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual.float(), expected, atol=5e-3, rtol=5e-3)
