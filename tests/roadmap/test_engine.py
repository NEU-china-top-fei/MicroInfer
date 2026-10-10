import pytest

from microinfer.config import SampleParameter
from microinfer.engine.contracts import (
    RequestOutput,
    ScheduledSequence,
    SchedulerOutput,
)
from microinfer.engine.llm_engine import LLMEngine

pytestmark = pytest.mark.roadmap


@pytest.mark.parametrize("fail", [False, True])
def test_schedule_execute_commit_order(fail):
    events = []
    work = SchedulerOutput((ScheduledSequence(8, (1,), 0, (0,), True),))
    expected = [RequestOutput(8, (2,), True, "length")]

    class SchedulerSpy:
        def schedule(self):
            events.append("schedule")
            return work

        def update(self, output, tokens):
            assert output is work and tokens == {8: 2}
            events.append("commit")
            return expected

    class RunnerSpy:
        def execute(self, output):
            assert output is work
            events.append("execute")
            if fail:
                raise RuntimeError("device failure")
            return {8: 2}

    engine = LLMEngine(SchedulerSpy(), RunnerSpy())
    if fail:
        with pytest.raises(RuntimeError, match="device failure"):
            engine.step()
        assert events == ["schedule", "execute"]
    else:
        assert engine.step() == expected
        assert events == ["schedule", "execute", "commit"]


def test_idle_engine_does_not_execute_or_commit():
    class IdleScheduler:
        def schedule(self):
            return SchedulerOutput()

        def update(self, *args):
            pytest.fail("idle work was committed")

    class IdleRunner:
        def execute(self, *args):
            pytest.fail("idle work was executed")

    assert LLMEngine(IdleScheduler(), IdleRunner()).step() == []


@pytest.mark.integration
def test_bounded_generation_loop_with_real_scheduler_runner_and_sampler(
    scheduler_factory, sequence_cls
):
    import torch
    from torch import nn
    from torch.nn import functional as F

    from microinfer.layer.sampler import Sampler
    from microinfer.runner import ModelRunner

    class TokenModel(nn.Module):
        def forward_batch(self, batch, kv_cache):
            assert not torch.is_grad_enabled()
            return F.one_hot((batch.input_ids + 1) % 16, 16).float()

    scheduler, manager = scheduler_factory(max_num_batched_tokens=3)
    engine = LLMEngine(
        scheduler, ModelRunner(TokenModel(), None, Sampler(), block_size=4, greedy=True)
    )
    a = sequence_cls([1, 2, 3, 4, 5], SampleParameter(max_tokens=3))
    b = sequence_cls([9], SampleParameter(max_tokens=3))
    for seq in [a, b]:
        engine.add_request(seq)
    finished = {}
    for _ in range(16):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step():
            if output.finished:
                assert output.seq_id not in finished
                finished[output.seq_id] = output.token_ids
    else:
        pytest.fail("generation loop made no bounded progress")
    assert finished == {a.seq_id: (6, 7, 8), b.seq_id: (10, 11, 12)}
    assert not manager.used_block_ids


def test_engine_abort_delegates(scheduler_factory, sequence_cls):
    scheduler, _ = scheduler_factory()
    engine = LLMEngine(scheduler, object())
    seq = sequence_cls([1])
    engine.add_request(seq)
    assert engine.has_unfinished_requests()
    assert engine.abort(seq.seq_id)
    assert not engine.has_unfinished_requests()
