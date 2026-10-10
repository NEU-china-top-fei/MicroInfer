import pytest
import torch
from torch import nn
from torch.nn import functional as F

from microinfer.engine.contracts import ScheduledSequence, SchedulerOutput
from microinfer.runner import ModelRunner

pytestmark = pytest.mark.roadmap


class ToyModel(nn.Module):
    """Control-flow oracle only; real model/cache numerics live in test_model_cache."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_batch(self, batch, kv_cache):
        assert not torch.is_grad_enabled() and not self.training
        self.calls.append((batch, kv_cache))
        return F.one_hot((batch.input_ids + 1) % 16, 16).float()


class ArgmaxSampler:
    def __init__(self):
        self.calls = []

    def sample(self, logits, temperatures, *, generator=None, greedy=False):
        self.calls.append((logits.clone(), temperatures.clone(), greedy))
        return logits.argmax(-1)


def test_runner_samples_only_completed_rows_and_maps_ids():
    model, sampler, cache = ToyModel(), ArgmaxSampler(), object()
    runner = ModelRunner(model, cache, sampler, block_size=4, greedy=True)
    work = SchedulerOutput(
        (
            ScheduledSequence(91, (1, 2), 0, (2,), False),
            ScheduledSequence(17, (7,), 4, (0, 1), True, 0.7),
        )
    )
    assert runner.execute(work) == {17: 8}
    assert len(model.calls) == 1 and model.calls[0][1] is cache
    assert sampler.calls[0][0].shape == (1, 16)
    torch.testing.assert_close(sampler.calls[0][1], torch.tensor([0.7]))
    assert sampler.calls[0][2] is True
    assert work.items[0].start_pos == 0


def test_empty_work_skips_model_and_sampler():
    model, sampler = ToyModel(), ArgmaxSampler()
    assert (
        ModelRunner(model, None, sampler, block_size=4).execute(SchedulerOutput()) == {}
    )
    assert not model.calls and not sampler.calls


def test_incomplete_chunk_executes_model_without_sampling():
    model, sampler = ToyModel(), ArgmaxSampler()
    work = SchedulerOutput((ScheduledSequence(1, (1, 2), 0, (0,), False),))
    assert ModelRunner(model, None, sampler, block_size=4).execute(work) == {}
    assert len(model.calls) == 1 and not sampler.calls


def test_runner_propagates_model_failure():
    class BrokenModel(nn.Module):
        def forward_batch(self, batch, kv_cache):
            raise RuntimeError("execution failed")

    sampler = ArgmaxSampler()
    work = SchedulerOutput((ScheduledSequence(1, (1,), 0, (0,), True),))
    with pytest.raises(RuntimeError, match="execution failed"):
        ModelRunner(BrokenModel(), None, sampler, block_size=4).execute(work)
    assert not sampler.calls
