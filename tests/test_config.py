import pytest

from microinfer.config import QwenConfig, SampleParameter


def test_config_overrides_are_independent(tiny_config):
    assert tiny_config.hidden_dim == 16
    assert QwenConfig().hidden_dim == 1024
    assert tiny_config.num_q_head % tiny_config.num_kv_head == 0


@pytest.mark.parametrize("temperature", [0.1, 1.0, 2.0])
def test_sampling_parameters(temperature):
    params = SampleParameter(temperature=temperature, max_tokens=7, ignore_eos=True)
    assert (params.temperature, params.max_tokens, params.ignore_eos) == (
        temperature,
        7,
        True,
    )


@pytest.mark.parametrize("temperature", [0.0, -1.0, 1e-11])
def test_sampling_rejects_unsupported_temperature(temperature):
    with pytest.raises(AssertionError, match="greedy sampling"):
        SampleParameter(temperature=temperature)


def test_config_supplies_attention_rope_base():
    assert QwenConfig().rope_base > 0
