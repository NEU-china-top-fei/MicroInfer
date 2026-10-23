from dataclasses import FrozenInstanceError

import pytest

from microinfer.config.engine import EngineConfig
from microinfer.engine.contracts import ScheduledSequence, SchedulerOutput


@pytest.mark.parametrize(
    "field", ["max_num_seqs", "max_num_batched_tokens", "max_model_len"]
)
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_limits_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        EngineConfig(**{field: value})


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_eos_rejects_invalid_ids(value):
    with pytest.raises(ValueError, match="eos_token_id"):
        EngineConfig(eos_token_id=value)


@pytest.mark.parametrize("field", ["debug_mode", "use_compile"])
@pytest.mark.parametrize("value", [0, 1, 1.0, "true", None])
def test_bool_flags_reject_non_bool(field, value):
    with pytest.raises(ValueError, match=field):
        EngineConfig(**{field: value})


@pytest.mark.parametrize("field", ["debug_mode", "use_compile"])
def test_bool_flags_accept_bool(field):
    config = EngineConfig(**{field: True})
    assert getattr(config, field) is True


def test_snapshot_counts_queries_not_cached_prefix():
    item = ScheduledSequence(7, (4, 5), 3, (2, 0), True)
    output = SchedulerOutput((item,))
    assert item.context_len == 5
    assert output.num_scheduled_tokens == 2
    with pytest.raises(FrozenInstanceError):
        item.start_pos = 9
    assert SchedulerOutput().num_scheduled_tokens == 0
