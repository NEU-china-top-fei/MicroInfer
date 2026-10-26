import pytest

from microinfer.config.engine import EngineConfig


@pytest.mark.parametrize(
    "field",
    [
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_num_prefill_tokens",
        "max_model_len",
    ],
)
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_limits_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        EngineConfig(**{field: value})


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_eos_rejects_invalid_ids(value):
    with pytest.raises(ValueError, match="eos_token_id"):
        EngineConfig(eos_token_id=value)


@pytest.mark.parametrize(
    "field", ["debug_mode", "use_compile", "use_cuda_graph", "mixed_batch"]
)
@pytest.mark.parametrize("value", [0, 1, 1.0, "true", None])
def test_bool_flags_reject_non_bool(field, value):
    with pytest.raises(ValueError, match=field):
        EngineConfig(**{field: value})


@pytest.mark.parametrize(
    "field", ["debug_mode", "use_compile", "use_cuda_graph", "mixed_batch"]
)
def test_bool_flags_accept_bool(field):
    config = EngineConfig(**{field: True})
    assert getattr(config, field) is True


def test_compile_and_cuda_graph_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        EngineConfig(use_compile=True, use_cuda_graph=True)
