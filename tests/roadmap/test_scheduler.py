import pytest

from microinfer.config import SampleParameter
from microinfer.engine.sequence import SequenceStatus

pytestmark = pytest.mark.roadmap


def test_empty_scheduler(scheduler_factory):
    scheduler, manager = scheduler_factory()
    assert not scheduler.has_unfinished_requests()
    assert scheduler.schedule().items == ()
    assert not manager.used_block_ids


def test_admission_is_lazy_and_rejects_duplicate(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory()
    seq = sequence_cls([1, 2])
    scheduler.add(seq)
    assert scheduler.has_unfinished_requests()
    assert not seq.block_table
    assert not manager.used_block_ids
    with pytest.raises(ValueError):
        scheduler.add(seq)


@pytest.mark.parametrize("prompt,max_tokens", [([1] * 32, 2), ([1], 0)])
def test_invalid_request_rejected(scheduler_factory, sequence_cls, prompt, max_tokens):
    scheduler, manager = scheduler_factory()
    with pytest.raises(ValueError):
        scheduler.add(sequence_cls(prompt, SampleParameter(max_tokens=max_tokens)))
    assert not manager.used_block_ids


def test_prompt_larger_than_entire_pool_rejected(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory(num_blocks=1)
    with pytest.raises(ValueError):
        scheduler.add(sequence_cls([1, 2, 3, 4, 5]))
    assert not manager.used_block_ids


def test_sequence_and_manager_block_sizes_must_agree(sequence_cls):
    from microinfer.config.engine import EngineConfig
    from microinfer.engine.blockmanager import BlockManager
    from microinfer.engine.scheduler import Scheduler

    manager = BlockManager(4, 8)
    with pytest.raises(ValueError):
        scheduler = Scheduler(EngineConfig(), manager)
        scheduler.add(sequence_cls([1, 2]))
    assert not manager.used_block_ids


def test_chunked_prefill_decode_and_release(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory(max_num_batched_tokens=3)
    seq = sequence_cls([1, 2, 3, 4, 5], SampleParameter(max_tokens=2))
    scheduler.add(seq)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == 3
    assert first.items[0].token_ids == (1, 2, 3)
    assert not first.items[0].sample
    assert seq.num_cached_tokens == 0
    assert not manager.hash_to_block_id
    assert scheduler.update(first, {}) == []
    assert seq.num_cached_tokens == 3
    assert seq.num_completion_tokens == 0
    second = scheduler.schedule()
    assert second.items[0].start_pos == 3
    assert second.items[0].token_ids == (4, 5)
    assert second.items[0].sample
    outputs = scheduler.update(second, {seq.seq_id: 6})
    assert outputs[0].token_ids == (6,)
    assert not outputs[0].finished
    assert seq.num_cached_tokens == 5
    assert seq.num_scheduled_tokens == 0
    assert manager.compute_hash([1, 2, 3, 4]) in manager.hash_to_block_id
    third = scheduler.schedule()
    assert third.items[0].token_ids == (6,)
    assert third.items[0].start_pos == 5
    done = scheduler.update(third, {seq.seq_id: 7})[0]
    assert (done.token_ids, done.finished, done.finish_reason) == (
        (6, 7),
        True,
        "length",
    )
    assert seq.status == SequenceStatus.FINISHED
    assert not seq.block_table and not manager.used_block_ids
    assert not scheduler.has_unfinished_requests()


def test_running_then_waiting_share_token_budget(scheduler_factory, sequence_cls):
    scheduler, _ = scheduler_factory(max_num_batched_tokens=4)
    a = sequence_cls([1])
    scheduler.add(a)
    scheduler.update(scheduler.schedule(), {a.seq_id: 2})
    b = sequence_cls([5, 6, 7, 8, 9])
    scheduler.add(b)
    batch = scheduler.schedule()
    assert [i.seq_id for i in batch.items] == [a.seq_id, b.seq_id]
    assert [i.token_ids for i in batch.items] == [(2,), (5, 6, 7)]
    assert [i.sample for i in batch.items] == [True, False]
    assert batch.num_scheduled_tokens == 4


def test_max_sequences_and_fcfs(scheduler_factory, sequence_cls):
    scheduler, _ = scheduler_factory(max_num_seqs=1)
    a, b = [sequence_cls([i], SampleParameter(max_tokens=1)) for i in [1, 2]]
    scheduler.add(a)
    scheduler.add(b)
    first = scheduler.schedule()
    assert [i.seq_id for i in first.items] == [a.seq_id]
    scheduler.update(first, {a.seq_id: 3})
    assert [i.seq_id for i in scheduler.schedule().items] == [b.seq_id]


def test_completed_prefix_reused_by_next_request(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory()
    a = sequence_cls([1, 2, 3, 4, 5], SampleParameter(max_tokens=1))
    scheduler.add(a)
    first = scheduler.schedule()
    old_block = first.items[0].block_table[0]
    scheduler.update(first, {a.seq_id: 9})
    b = sequence_cls([1, 2, 3, 4, 6])
    scheduler.add(b)
    second = scheduler.schedule().items[0]
    assert (second.start_pos, second.token_ids) == (4, (6,))
    assert second.block_table[0] == old_block
    assert manager.blocks[old_block].ref_count == 1


@pytest.mark.parametrize(
    "ignore_eos,reason,finished", [(False, "eos", True), (True, None, False)]
)
def test_eos_policy(scheduler_factory, sequence_cls, ignore_eos, reason, finished):
    scheduler, _ = scheduler_factory()
    seq = sequence_cls([1], SampleParameter(ignore_eos=ignore_eos))
    scheduler.add(seq)
    output = scheduler.update(scheduler.schedule(), {seq.seq_id: 0})[0]
    assert (output.finished, output.finish_reason) == (finished, reason)
    assert output.token_ids == (0,)


def test_context_limit_finishes_without_out_of_bounds_slot(
    scheduler_factory, sequence_cls
):
    scheduler, manager = scheduler_factory(max_model_len=4)
    seq = sequence_cls([1, 2, 3])
    scheduler.add(seq)
    output = scheduler.update(scheduler.schedule(), {seq.seq_id: 4})[0]
    assert output.finished and output.finish_reason == "length"
    assert not manager.used_block_ids


def test_sample_ids_validated_before_any_commit(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory()
    seq = sequence_cls([1, 2, 3, 4])
    scheduler.add(seq)
    output = scheduler.schedule()
    for wrong in [{}, {seq.seq_id + 1000: 8}]:
        with pytest.raises(ValueError):
            scheduler.update(output, wrong)
        assert seq.num_cached_tokens == 0 and seq.token_ids == [1, 2, 3, 4]
        assert not manager.hash_to_block_id
    scheduler.update(output, {seq.seq_id: 8})
    with pytest.raises(ValueError):
        scheduler.update(output, {seq.seq_id: 8})


def test_cannot_schedule_or_abort_in_flight(scheduler_factory, sequence_cls):
    scheduler, _ = scheduler_factory()
    seq = sequence_cls([1])
    scheduler.add(seq)
    scheduler.schedule()
    with pytest.raises(RuntimeError) as scheduling_error:
        scheduler.schedule()
    assert not isinstance(scheduling_error.value, NotImplementedError)
    with pytest.raises(RuntimeError) as abort_error:
        scheduler.abort(seq.seq_id)
    assert not isinstance(abort_error.value, NotImplementedError)


@pytest.mark.parametrize("run_first", [False, True])
def test_abort_releases_resources(scheduler_factory, sequence_cls, run_first):
    scheduler, manager = scheduler_factory()
    seq = sequence_cls([1, 2, 3])
    scheduler.add(seq)
    if run_first:
        scheduler.update(scheduler.schedule(), {seq.seq_id: 4})
    assert scheduler.abort(seq.seq_id)
    assert not scheduler.abort(seq.seq_id)
    assert not scheduler.has_unfinished_requests()
    assert not manager.used_block_ids and not seq.block_table


def test_preemption_preserves_generated_tokens(scheduler_factory, sequence_cls):
    scheduler, _ = scheduler_factory()
    seq = sequence_cls([1, 2, 3, 4, 5])
    scheduler.add(seq)
    scheduler.update(scheduler.schedule(), {seq.seq_id: 6})
    scheduler.preempt(seq.seq_id)
    assert seq.token_ids == [1, 2, 3, 4, 5, 6]
    assert seq.num_cached_tokens == seq.num_scheduled_tokens == 0
    assert seq.status == SequenceStatus.WAITING and seq.is_prefill
    assert not seq.block_table
    next_batch = scheduler.schedule()
    scheduler.update(next_batch, {seq.seq_id: 7})
    assert seq.completion_token_ids == [6, 7]


def test_decode_pressure_preempts_newest_other_request(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory(num_blocks=2)
    a = sequence_cls([1, 2, 3, 4], SampleParameter(max_tokens=2))
    b = sequence_cls([7, 8, 9, 10], SampleParameter(max_tokens=2))
    scheduler.add(a)
    scheduler.add(b)
    scheduler.update(scheduler.schedule(), {a.seq_id: 5, b.seq_id: 11})
    batch = scheduler.schedule()
    assert [i.seq_id for i in batch.items] == [a.seq_id]
    assert b.status == SequenceStatus.WAITING and not b.block_table
    scheduler.update(batch, {a.seq_id: 6})
    scheduler.update(scheduler.schedule(), {b.seq_id: 12})
    assert b.completion_token_ids == [11, 12]
    assert not manager.used_block_ids


def test_unfit_single_request_errors_instead_of_deadlocking(
    scheduler_factory, sequence_cls
):
    scheduler, _ = scheduler_factory(num_blocks=1)
    seq = sequence_cls([1, 2, 3, 4])
    scheduler.add(seq)
    scheduler.update(scheduler.schedule(), {seq.seq_id: 5})
    with pytest.raises(RuntimeError) as error:
        scheduler.schedule()
    assert not isinstance(error.value, NotImplementedError)
