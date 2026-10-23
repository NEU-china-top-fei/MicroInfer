import pytest
import torch

from microinfer.layer.norm import FuseLayerNorm


def rms_reference(x, weight, eps):
    work = x.float()
    return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + eps)).to(
        x.dtype
    ) * weight


@pytest.mark.parametrize("with_residual", [False, True])
def test_norm_matches_rms_formula(tiny_config, seeded_torch, device, with_residual):
    layer = FuseLayerNorm(tiny_config).to(device)
    with torch.no_grad():
        layer.norm.weight.copy_(
            torch.linspace(0.5, 1.5, tiny_config.hidden_dim, device=device)
        )
    x = torch.randn(2, 3, tiny_config.hidden_dim).to(device)
    residual = torch.randn_like(x) if with_residual else None
    before = x.clone()
    actual, carried = layer(x, residual)
    combined = x if residual is None else x + residual
    torch.testing.assert_close(
        actual, rms_reference(combined, layer.norm.weight, tiny_config.norm_eps)
    )
    torch.testing.assert_close(carried, combined)
    torch.testing.assert_close(x, before)


def test_norm_returns_accumulated_residual(tiny_config):
    layer = FuseLayerNorm(tiny_config)
    x = torch.ones(2, 3, tiny_config.hidden_dim)
    residual = torch.full_like(x, 2.0)
    _, carried = layer(x, residual)
    torch.testing.assert_close(carried, x + residual)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_norm_supports_half_precision(
    tiny_config, seeded_torch, device, dtype, with_residual
):
    layer = FuseLayerNorm(tiny_config).to(device=device, dtype=dtype)
    x = torch.randn(2, 3, tiny_config.hidden_dim).to(device=device, dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None
    combined = x if residual is None else x + residual
    actual, carried = layer(x, residual)
    assert actual.dtype == dtype
    torch.testing.assert_close(carried, combined)
    torch.testing.assert_close(
        actual.float(),
        rms_reference(
            combined, torch.ones_like(combined), tiny_config.norm_eps
        ).float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_zero_input_stays_finite(tiny_config):
    actual, _ = FuseLayerNorm(tiny_config)(
        torch.zeros(2, tiny_config.hidden_dim)
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual))
