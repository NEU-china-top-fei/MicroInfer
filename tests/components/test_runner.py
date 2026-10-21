import pytest
import torch
from torch import nn
from torch.nn import functional as F

from microinfer.runner import ModelRunner
from microinfer.utils.context import get_context, reset_context

pytestmark = [pytest.mark.component, pytest.mark.nano]


@pytest.fixture(autouse=True)
def clear_context():
    reset_context()
    yield
    reset_context()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, positions):
        assert not torch.is_grad_enabled()
        self.calls.append((input_ids.clone(), positions.clone()))
        return F.one_hot((input_ids + 1) % 16, 16).float()

    def compute_logits(self, hidden):
        context = get_context()
        return (
            hidden[context.cu_seqlens_q[1:].long() - 1]
            if context.is_prefill
            else hidden
        )


class ArgmaxSampler:
    def __init__(self):
        self.calls = []

    def __call__(self, logits, temperatures):
        self.calls.append((logits.clone(), temperatures.clone()))
        return logits.argmax(-1)


def test_prepare_prefill_with_prefix_cache(sequence_cls):
    a, b = sequence_cls([10, 11, 12, 13, 14, 15]), sequence_cls([24, 25])
    a.num_cached_tokens, a.num_scheduled_tokens, a.block_table = 3, 3, [2, 0]
    b.num_scheduled_tokens, b.block_table = 2, [5]
    runner = ModelRunner(None, None, None, block_size=4)
    ids, positions = runner.prepare_prefill([a, b])
    context = get_context()
    assert ids.tolist() == [13, 14, 15, 24, 25]
    assert positions.tolist() == [3, 4, 5, 0, 1]
    assert context.is_prefill
    assert context.cu_seqlens_q.tolist() == [0, 3, 5]
    assert context.cu_seqlens_k.tolist() == [0, 6, 8]
    assert (context.max_seqlen_q, context.max_seqlen_k) == (3, 6)
    assert context.slot_mapping.tolist() == [11, 0, 1, 20, 21]
    assert context.slot_mapping.dtype == torch.int32
    assert context.block_tables.tolist() == [[2, 0], [5, -1]]
    assert context.context_lens is None


def test_prepare_prefill_without_cache_needs_no_block_table(sequence_cls):
    seq = sequence_cls([1, 2])
    seq.num_scheduled_tokens, seq.block_table = 2, [0]
    ModelRunner(None, None, None, block_size=4).prepare_prefill([seq])
    context = get_context()
    assert context.block_tables is None
    assert context.cu_seqlens_k.tolist() == context.cu_seqlens_q.tolist() == [0, 2]


def test_prepare_decode_uses_one_last_token_per_sequence(sequence_cls):
    a, b = sequence_cls([1, 2, 3, 4, 5]), sequence_cls([7, 8])
    a.block_table, b.block_table = [2, 0], [1]
    a.num_cached_tokens, b.num_cached_tokens = 4, 1
    a.num_scheduled_tokens = b.num_scheduled_tokens = 1
    ids, positions = ModelRunner(None, None, None, block_size=4).prepare_decode([a, b])
    context = get_context()
    assert ids.tolist() == [5, 8] and positions.tolist() == [4, 1]
    assert not context.is_prefill
    assert context.context_lens.tolist() == [5, 2]
    assert context.slot_mapping.tolist() == [0, 5]
    assert context.block_tables.tolist() == [[2, 0], [1, -1]]
    assert context.cu_seqlens_q is context.cu_seqlens_k is None


def test_runner_returns_positional_samples_even_for_incomplete_chunk(sequence_cls):
    seq = sequence_cls([1, 2, 3, 4, 5])
    seq.block_table, seq.num_scheduled_tokens = [0, 1], 3
    sampler, model = ArgmaxSampler(), ToyModel()
    runner = ModelRunner(model, None, sampler, block_size=4)
    assert runner.run([seq], True) == [4]
    assert seq.token_ids == [1, 2, 3, 4, 5] and seq.num_cached_tokens == 0
    assert sampler.calls[0][0].shape == (1, 16)
    assert get_context().slot_mapping is None


def test_run_preserves_sequence_order(sequence_cls):
    a, b = sequence_cls([1, 2]), sequence_cls([7])
    a.block_table, b.block_table = [0], [1]
    a.num_scheduled_tokens, b.num_scheduled_tokens = 2, 1
    sampler = ArgmaxSampler()
    result = ModelRunner(ToyModel(), None, sampler, block_size=4).run([a, b], True)
    assert result == [3, 8]
    assert all(type(token) is int for token in result)
    assert sampler.calls[0][1].tolist() == [a.temperature, b.temperature]


@pytest.mark.extension
def test_empty_work_skips_model(sequence_cls):
    model, sampler = ToyModel(), ArgmaxSampler()
    assert ModelRunner(model, None, sampler, block_size=4).run([], True) == []
    assert not model.calls and not sampler.calls


@pytest.mark.extension
def test_exception_cleanup_clears_context(sequence_cls):
    class BrokenModel(ToyModel):
        def forward(self, ids, positions):
            raise RuntimeError("execution failed")

    seq = sequence_cls([1])
    seq.num_scheduled_tokens, seq.block_table = 1, [0]
    with pytest.raises(RuntimeError, match="execution failed"):
        ModelRunner(BrokenModel(), None, ArgmaxSampler(), block_size=4).run([seq], True)
    assert get_context().slot_mapping is None
