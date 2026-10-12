from dataclasses import replace

import pytest
import torch

from microinfer.engine.contracts import ScheduledSequence, SchedulerOutput
from microinfer.worker.batch import prepare_batch

pytestmark = [pytest.mark.roadmap, pytest.mark.extension]


def test_optional_packer_handles_distinct_cached_prefixes():
    a = ScheduledSequence(10, (13, 14, 15), 3, (2, 0), False)
    b = ScheduledSequence(20, (25,), 1, (5,), True, 0.5)
    snapshot = SchedulerOutput((a, b))
    batch = prepare_batch(snapshot, block_size=4)
    assert batch.input_ids.tolist() == [13, 14, 15, 25]
    assert batch.positions.tolist() == [3, 4, 5, 1]
    assert batch.slot_mapping.tolist() == [11, 0, 1, 21]
    assert batch.cu_seqlens_q.tolist() == [0, 3, 4]
    assert batch.context_lens.tolist() == [6, 2]
    assert batch.block_tables.tolist() == [[2, 0], [5, -1]]
    assert batch.sample_indices.tolist() == [3]
    assert batch.temperatures.tolist() == [0.5]
    assert batch.seq_ids == (10, 20) and batch.sample_seq_ids == (20,)
    assert snapshot.items == (a, b)
    for name in ["input_ids", "positions", "slot_mapping", "sample_indices"]:
        assert getattr(batch, name).dtype == torch.int64
    for name in ["cu_seqlens_q", "context_lens", "block_tables"]:
        assert getattr(batch, name).dtype == torch.int32


def test_sampling_indices_select_last_token_of_each_complete_request():
    batch = prepare_batch(
        SchedulerOutput(
            (
                ScheduledSequence(1, (1, 2), 0, (2,), True),
                ScheduledSequence(2, (3, 4, 5), 0, (0,), True),
            )
        ),
        4,
    )
    assert batch.sample_indices.tolist() == [1, 4]
    assert batch.sample_seq_ids == (1, 2)


def test_empty_batch_has_valid_offsets():
    batch = prepare_batch(SchedulerOutput(), 4)
    assert batch.input_ids.numel() == batch.sample_indices.numel() == 0
    assert batch.cu_seqlens_q.tolist() == [0]
    assert batch.block_tables.shape == (0, 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"start_pos": -1},
        {"token_ids": ()},
        {"block_table": ()},
        {"block_table": (-1,)},
        {"start_pos": 4},
        {"temperature": 0},
    ],
)
def test_invalid_batch_metadata_rejected(changes):
    item = replace(ScheduledSequence(1, (2,), 0, (0,), True), **changes)
    with pytest.raises(ValueError):
        prepare_batch(SchedulerOutput((item,)), 4)


def test_duplicate_request_ids_rejected():
    item = ScheduledSequence(1, (2,), 0, (0,), True)
    with pytest.raises(ValueError):
        prepare_batch(SchedulerOutput((item, item)), 4)
