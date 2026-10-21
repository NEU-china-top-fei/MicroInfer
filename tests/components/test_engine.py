import pytest

from microinfer.config import SampleParameter
from microinfer.engine.llm_engine import LLMEngine
from microinfer.engine.sequence import SequenceStatus

pytestmark = [pytest.mark.component, pytest.mark.nano]


@pytest.mark.parametrize("mode", [True, False])
def test_step_order_finished_outputs_and_signed_token_count(sequence_cls, mode):
    events = []
    seq = sequence_cls([1, 2, 3])
    if not mode:
        seq.append_token(4)
    seq.num_scheduled_tokens = 3 if mode else 1
    expected = seq.completion_token_ids + [5]

    class SchedulerSpy:
        def schedule(self):
            events.append("schedule")
            return [seq], mode

        def postprocess(self, seqs, tokens, is_prefill):
            assert seqs == [seq] and tokens == [5] and is_prefill == mode
            events.append("postprocess")
            seq.num_scheduled_tokens = 0
            seq.append_token(5)
            seq.status = SequenceStatus.FINISHED

    class RunnerSpy:
        def run(self, seqs, is_prefill):
            assert seqs == [seq] and is_prefill == mode
            events.append("run")
            return [5]

    outputs, count = LLMEngine(SchedulerSpy(), RunnerSpy()).step()
    assert outputs == [(seq.seq_id, expected)]
    assert count == (3 if mode else -1)
    assert events == ["schedule", "run", "postprocess"]


def test_execution_error_does_not_postprocess(sequence_cls):
    seq = sequence_cls([1])
    seq.num_scheduled_tokens = 1

    class SchedulerSpy:
        def schedule(self):
            return [seq], True

        def postprocess(self, *args):
            pytest.fail("failed KV was committed")

    class RunnerSpy:
        def run(self, *args):
            raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        LLMEngine(SchedulerSpy(), RunnerSpy()).step()


def test_unfinished_sequences_produce_no_output_event(sequence_cls):
    seq = sequence_cls([1, 2, 3])
    seq.num_scheduled_tokens = 1

    class SchedulerSpy:
        def schedule(self):
            return [seq], True

        def postprocess(self, seqs, tokens, is_prefill):
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0

    class RunnerSpy:
        def run(self, *args):
            return [99]

    assert LLMEngine(SchedulerSpy(), RunnerSpy()).step() == ([], 1)


@pytest.mark.integration
def test_generation_terminates_with_real_scheduler(scheduler_factory, sequence_cls):
    # Tests lifecycle, not model numerics. Runner/model are verified separately.
    class DeterministicRunner:
        def run(self, seqs, is_prefill):
            return [
                (s.token_ids[s.num_cached_tokens + s.num_scheduled_tokens - 1] + 1) % 16
                for s in seqs
            ]

    scheduler, manager = scheduler_factory(max_num_batched_tokens=3)
    engine = LLMEngine(scheduler, DeterministicRunner())
    a = sequence_cls([1, 2, 3, 4, 5], SampleParameter(max_tokens=3))
    b = sequence_cls([9], SampleParameter(max_tokens=3))
    for seq in [a, b]:
        engine.add_request(seq)
    finished = {}
    for _ in range(16):
        if engine.is_finished():
            break
        outputs, _ = engine.step()
        for seq_id, tokens in outputs:
            assert seq_id not in finished
            finished[seq_id] = tokens
    else:
        pytest.fail("generation did not terminate")
    assert finished == {a.seq_id: [6, 7, 8], b.seq_id: [10, 11, 12]}
    assert not manager.used_block_ids
