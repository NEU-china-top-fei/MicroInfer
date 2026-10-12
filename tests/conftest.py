import pytest


@pytest.fixture
def sequence_cls(monkeypatch):
    from microinfer.engine.sequence import Sequence

    # Keep block boundaries small and restore the class attribute after each test.
    monkeypatch.setattr(Sequence, "block_size", 4)
    return Sequence


@pytest.fixture
def tiny_config():
    from microinfer.config import QwenConfig

    return QwenConfig(
        vocab_size=32,
        hidden_dim=16,
        MLP_inter_dim=24,
        num_q_head=4,
        num_kv_head=2,
        head_dim=4,
        num_hidden_layer=2,
        max_pos_embed=16,
        fused_layer_norm=False,
    )


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    import torch

    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    return (
        torch.device("cuda", torch.cuda.current_device())
        if request.param == "cuda"
        else torch.device("cpu")
    )


@pytest.fixture
def seeded_torch():
    import torch

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        yield torch


@pytest.fixture
def eager_norm(monkeypatch):
    from microinfer.layer.norm import FuseLayerNorm

    # Validate arithmetic without compiler startup; no arithmetic is mocked.
    method = FuseLayerNorm.fwd_fuse_tobe_compile
    monkeypatch.setattr(
        FuseLayerNorm, "fwd_fuse_tobe_compile", getattr(method, "__wrapped__", method)
    )
    return FuseLayerNorm
