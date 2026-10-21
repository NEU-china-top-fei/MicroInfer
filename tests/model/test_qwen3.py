import pytest
import torch
from torch import nn


def test_model_constructs_registered_layers(tiny_config):
    # Import in the test: a model construction bug must not block engine tests.
    from microinfer.model.qwen3 import QwenModel

    tiny_config.rope_base = 10000.0
    model = QwenModel(tiny_config)
    assert isinstance(model, nn.Module)
    assert len(model.transformer_layers) == tiny_config.num_hidden_layer
    assert model.embedding.weight.shape == (32, 16)
    assert len(list(model.parameters())) > 2


def test_embedding_final_norm_and_logits_without_transformer(tiny_config, seeded_torch):
    from microinfer.model.qwen3 import QwenModel

    # Zero layers isolates embedding / final RMSNorm / LM head from attention.
    tiny_config.num_hidden_layer = 0
    model = QwenModel(tiny_config).eval()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    embedded = model.embedding(ids)
    expected_hidden = embedded * torch.rsqrt(
        embedded.square().mean(-1, keepdim=True) + tiny_config.norm_eps
    )
    hidden = model(ids, torch.arange(3))
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == (2, 3, 16)
    torch.testing.assert_close(hidden, expected_hidden)
    logits = model.compute_logits(hidden)
    expected_logits = hidden @ model.out_logits.weight.T
    if model.out_logits.bias is not None:
        expected_logits = expected_logits + model.out_logits.bias
    assert logits.shape == (2, 3, 32)
    torch.testing.assert_close(logits, expected_logits)


def test_model_passes_positions_and_accumulates_final_residual(tiny_config):
    from microinfer.model.qwen3 import QwenModel

    class ProbeLayer(nn.Module):
        def forward(self, hidden, position, residual=None):
            assert torch.equal(position, torch.arange(3))
            assert residual is None
            return hidden * 2, hidden.flip(-1)

    tiny_config.num_hidden_layer = 0
    model = QwenModel(tiny_config)
    model.transformer_layers = nn.ModuleList([ProbeLayer()])
    ids = torch.tensor([[1, 2, 3]])
    embedded = model.embedding(ids)
    combined = embedded * 2 + embedded.flip(-1)
    expected = combined * torch.rsqrt(
        combined.square().mean(-1, keepdim=True) + tiny_config.norm_eps
    )
    actual = model(ids, torch.arange(3))
    torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.integration
def test_real_model_prefill_is_causal(tiny_config, seeded_torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    from microinfer.model.qwen3 import QwenModel

    tiny_config.rope_base = 10000.0
    tiny_config.head_dim, tiny_config.hidden_dim = 32, 128
    model = QwenModel(tiny_config).cuda().half().eval()
    # Different future tokens must not change the shared prefix's logits.
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 7, 8]], device="cuda")
    with torch.inference_mode():
        hidden = model(ids, torch.arange(4, device="cuda"))
        logits = model.compute_logits(hidden)
    assert hidden.shape == (2, 4, 128)
    assert logits.shape == (2, 4, tiny_config.vocab_size)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits[0, :2], logits[1, :2], atol=5e-3, rtol=5e-3)
