"""Behavioral parity with nano-vllm bb823b3; no extra admission/state guards."""

import pytest

from microinfer.config import SampleParameter
from microinfer.engine.sequence import SequenceStatus

pytestmark = [pytest.mark.roadmap, pytest.mark.nano]


def test_empty_scheduler_is_finished(scheduler_factory):
    scheduler, _ = scheduler_factory()
    assert scheduler.is_finished()


def test_admission_is_lazy(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory()
    seq = sequence_cls([1, 2])
    scheduler.add(seq)
    assert not scheduler.is_finished()
    assert not seq.block_table and not manager.used_block_ids


def test_chunked_prefill_discards_intermediate_sample(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory(max_num_batched_tokens=3)
    seq = sequence_cls([1, 2, 3, 4, 5], SampleParameter(max_tokens=2))
    scheduler.add(seq)
    seqs, mode = scheduler.schedule()
    assert mode is True and seqs == [seq]
    assert seq.num_scheduled_tokens == 3 and seq.num_cached_tokens == 0
    assert seq.status == SequenceStatus.WAITING
    assert not manager.hash_to_block_id
    scheduler.postprocess(seqs, [99], mode)
    assert seq.token_ids == [1, 2, 3, 4, 5]  # 99 was sampled, but discarded.
    assert seq.num_cached_tokens == 3 and seq.num_scheduled_tokens == 0
    seqs, mode = scheduler.schedule()
    assert mode and seq.num_scheduled_tokens == 2
    assert seq.status == SequenceStatus.RUNNING
    scheduler.postprocess(seqs, [6], mode)
    assert seq.completion_token_ids == [6] and seq.num_cached_tokens == 5
    assert manager.compute_hash([1, 2, 3, 4]) in manager.hash_to_block_id
    seqs, mode = scheduler.schedule()
    assert mode is False and seq.num_scheduled_tokens == 1
    # breakpoint()
    scheduler.postprocess(seqs, [7], mode)
    assert seq.is_finished and seq.completion_token_ids == [6, 7]
    assert not manager.used_block_ids and scheduler.is_finished()


def test_new_prefill_has_priority_over_running_decode(scheduler_factory, sequence_cls):
    scheduler, _ = scheduler_factory(max_num_batched_tokens=4)
    a = sequence_cls([1])
    scheduler.add(a)
    scheduler.postprocess(*_step_args(scheduler, [2]))
    b = sequence_cls([5, 6])
    scheduler.add(b)
    seqs, mode = scheduler.schedule()
    assert mode is True and seqs == [b]
    assert a.num_cached_tokens == 1 and a.num_scheduled_tokens == 0


def _step_args(scheduler, tokens):
    seqs, mode = scheduler.schedule()
    return seqs, tokens, mode


def test_only_first_prefill_may_be_chunked(scheduler_factory, sequence_cls):
    scheduler, _ = scheduler_factory(max_num_batched_tokens=4)
    a, b = sequence_cls([1, 2]), sequence_cls([3, 4, 5])
    scheduler.add(a)
    scheduler.add(b)
    seqs, mode = scheduler.schedule()
    assert mode and seqs == [a]  # Two spare tokens do not split the second request.
    assert not b.block_table and b.num_scheduled_tokens == 0


def test_max_sequences_and_prefill_fcfs(scheduler_factory, sequence_cls):
    scheduler, _ = scheduler_factory(max_num_seqs=1)
    a, b = [sequence_cls([i], SampleParameter(max_tokens=1)) for i in [1, 2]]
    scheduler.add(a)
    scheduler.add(b)
    seqs, mode = scheduler.schedule()
    assert seqs == [a] and mode
    scheduler.postprocess(seqs, [3], mode)
    assert scheduler.schedule()[0] == [b]


def test_decode_uses_sequence_limit_not_prefill_token_budget(
    scheduler_factory, sequence_cls
):
    scheduler, _ = scheduler_factory(max_num_batched_tokens=1, max_num_seqs=2)
    a, b = sequence_cls([1]), sequence_cls([3])
    scheduler.add(a)
    scheduler.add(b)
    scheduler.postprocess(*_step_args(scheduler, [2]))
    scheduler.postprocess(*_step_args(scheduler, [4]))
    seqs, mode = scheduler.schedule()
    assert not mode and seqs == [a, b]
    assert sum(s.num_scheduled_tokens for s in seqs) == 2
    assert all(not s.is_prefill for s in seqs)


def test_completed_prefix_reused(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory()
    a = sequence_cls([1, 2, 3, 4, 5], SampleParameter(max_tokens=1))
    scheduler.add(a)
    seqs, mode = scheduler.schedule()
    # breakpoint()
    old_block = a.block_table[0]
    scheduler.postprocess(seqs, [9], mode)
    b = sequence_cls([1, 2, 3, 4, 6])
    scheduler.add(b)
    # breakpoint()
    seqs, mode = scheduler.schedule()
    assert mode and seqs == [b]
    assert b.num_cached_tokens == 4 and b.num_scheduled_tokens == 1
    assert b.block_table[0] == old_block
    assert manager.blocks[old_block].ref_count == 1


@pytest.mark.parametrize("ignore_eos,finished", [(False, True), (True, False)])
def test_eos_policy(scheduler_factory, sequence_cls, ignore_eos, finished):
    scheduler, _ = scheduler_factory()
    seq = sequence_cls([1], SampleParameter(ignore_eos=ignore_eos))
    scheduler.add(seq)
    scheduler.postprocess(*_step_args(scheduler, [0]))
    assert seq.is_finished == finished and seq.completion_token_ids == [0]


def test_preempt_preserves_tokens_and_requeues_at_front(
    scheduler_factory, sequence_cls
):
    scheduler, _ = scheduler_factory()
    seq = sequence_cls([1, 2, 3, 4, 5])
    scheduler.add(seq)
    scheduler.postprocess(*_step_args(scheduler, [6]))
    scheduler.running.remove(seq)  # nano caller owns removal before preempt().
    scheduler.preempt(seq)
    assert seq.token_ids == [1, 2, 3, 4, 5, 6]
    assert seq.num_cached_tokens == 0 and not seq.block_table
    assert seq.status == SequenceStatus.WAITING and seq.is_prefill
    assert scheduler.waiting[0] is seq
    scheduler.postprocess(*_step_args(scheduler, [7]))
    assert seq.completion_token_ids == [6, 7]


def test_decode_pressure_preempts_tail(scheduler_factory, sequence_cls):
    scheduler, manager = scheduler_factory(num_blocks=2)
    a = sequence_cls([1, 2, 3, 4], SampleParameter(max_tokens=2))
    b = sequence_cls([7, 8, 9, 10], SampleParameter(max_tokens=2))
    scheduler.add(a)
    scheduler.add(b)
    scheduler.postprocess(*_step_args(scheduler, [5, 11]))
    seqs, mode = scheduler.schedule()
    assert not mode and seqs == [a]
    assert b.status == SequenceStatus.WAITING and not b.block_table
    scheduler.postprocess(seqs, [6], mode)
    scheduler.postprocess(*_step_args(scheduler, [12]))
    assert b.completion_token_ids == [11, 12]
    assert scheduler.is_finished() and not manager.used_block_ids
