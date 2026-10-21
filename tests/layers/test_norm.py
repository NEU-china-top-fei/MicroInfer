import pytest
import torch


def rms_reference(x, weight, eps):
    work = x.float()
    return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + eps)).to(
        x.dtype
    ) * weight


@pytest.mark.parametrize("with_residual", [False, True])
def test_unfused_norm_matches_rms_formula(
    tiny_config, eager_norm, seeded_torch, device, with_residual
):
    layer = eager_norm(tiny_config).to(device)
    with torch.no_grad():
        layer.norm.weight.copy_(
            torch.linspace(0.5, 1.5, tiny_config.hidden_dim, device=device)
        )
    x = torch.randn(2, 3, tiny_config.hidden_dim).to(device)
    residual = torch.randn_like(x) if with_residual else None
    before = x.clone()
    actual, _ = layer(x, residual)
    combined = x if residual is None else x + residual
    torch.testing.assert_close(
        actual, rms_reference(combined, layer.norm.weight, tiny_config.norm_eps)
    )
    torch.testing.assert_close(x, before)


def test_norm_returns_accumulated_residual(tiny_config, eager_norm):
    x = torch.ones(2, 3, tiny_config.hidden_dim)
    residual = torch.full_like(x, 2.0)
    _, carried = eager_norm(tiny_config)(x, residual)
    torch.testing.assert_close(carried, x + residual)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_fused_norm_matches_rms_and_residual(
    tiny_config, eager_norm, seeded_torch, device, dtype, with_residual
):
    tiny_config.fused_layer_norm = True
    layer = eager_norm(tiny_config).to(device=device, dtype=dtype)
    x = torch.randn(2, 3, tiny_config.hidden_dim).to(device=device, dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None
    combined = x.float() if residual is None else x.float() + residual.float()
    actual, carried = layer(x, residual)
    assert actual.dtype == dtype
    torch.testing.assert_close(carried, combined.to(dtype))
    # Initial RMS weight is one; nonuniform weights are tested on unfused norm.
    torch.testing.assert_close(
        actual,
        rms_reference(combined, torch.ones_like(combined), tiny_config.norm_eps).to(
            dtype
        ),
    )


def test_zero_input_stays_finite(tiny_config, eager_norm):
    actual, _ = eager_norm(tiny_config)(torch.zeros(2, tiny_config.hidden_dim))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual))
