import pytest
import torch

from microinfer.utils.context import reset_context, set_context

pytestmark = [pytest.mark.component, pytest.mark.nano]


@pytest.mark.parametrize("is_prefill", [True, False])
def test_lm_head_selects_one_row_per_sequence(tiny_config, seeded_torch, is_prefill):
    from microinfer.model.qwen3 import QwenModel

    tiny_config.rope_base = 10000.0
    model = QwenModel(tiny_config).eval()
    # Two requests with query lengths 3 and 2. Decode has one row per request.
    hidden = torch.randn(5 if is_prefill else 2, tiny_config.hidden_dim)
    try:
        set_context(
            is_prefill,
            cu_seqlens_q=torch.tensor([0, 3, 5]) if is_prefill else None,
        )
        with torch.inference_mode():
            expected = model.out_logits(hidden[[2, 4]] if is_prefill else hidden)
            actual = model.compute_logits(hidden)
        assert actual.shape == (2, tiny_config.vocab_size)
        torch.testing.assert_close(actual, expected)
    finally:
        reset_context()
