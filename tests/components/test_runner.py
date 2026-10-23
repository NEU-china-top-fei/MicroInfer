import pytest
import torch
from torch import nn
from torch.nn import functional as F

from microinfer.runner import ModelRunner

pytestmark = [pytest.mark.component, pytest.mark.nano]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, positions, metadata=None):
        assert not torch.is_grad_enabled()
        self.calls.append((input_ids.clone(), positions.clone()))
        return F.one_hot((input_ids + 1) % 16, 16).float()

    def compute_logits(self, hidden, metadata=None):
        if metadata is not None and metadata.is_prefill:
            return hidden[metadata.cu_seqlens_q[1:].long() - 1]
        return hidden


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
    ids, positions, metadata = runner.prepare_prefill([a, b])
    assert ids.tolist() == [13, 14, 15, 24, 25]
    assert positions.tolist() == [3, 4, 5, 0, 1]
    assert metadata.is_prefill
    assert metadata.cu_seqlens_q.tolist() == [0, 3, 5]
    assert metadata.cu_seqlens_k.tolist() == [0, 6, 8]
    assert (metadata.max_seqlen_q, metadata.max_seqlen_k) == (3, 6)
    assert metadata.slot_mapping.tolist() == [11, 0, 1, 20, 21]
    assert metadata.slot_mapping.dtype == torch.int32
    assert metadata.block_tables.tolist() == [[2, 0], [5, -1]]
    assert metadata.context_lens is None


def test_prepare_prefill_without_cache_needs_no_block_table(sequence_cls):
    seq = sequence_cls([1, 2])
    seq.num_scheduled_tokens, seq.block_table = 2, [0]
    _, _, metadata = ModelRunner(None, None, None, block_size=4).prepare_prefill([seq])
    assert metadata.block_tables is None
    assert metadata.cu_seqlens_k.tolist() == metadata.cu_seqlens_q.tolist() == [0, 2]


def test_prepare_decode_uses_one_last_token_per_sequence(sequence_cls):
    a, b = sequence_cls([1, 2, 3, 4, 5]), sequence_cls([7, 8])
    a.block_table, b.block_table = [2, 0], [1]
    a.num_cached_tokens, b.num_cached_tokens = 4, 1
    a.num_scheduled_tokens = b.num_scheduled_tokens = 1
    ids, positions, metadata = ModelRunner(
        None, None, None, block_size=4
    ).prepare_decode([a, b])
    assert ids.tolist() == [5, 8] and positions.tolist() == [4, 1]
    assert not metadata.is_prefill
    assert metadata.context_lens.tolist() == [5, 2]
    assert metadata.slot_mapping.tolist() == [0, 5]
    assert metadata.block_tables.tolist() == [[2, 0], [1, -1]]
    assert metadata.cu_seqlens_q is metadata.cu_seqlens_k is None


def test_runner_returns_positional_samples_even_for_incomplete_chunk(sequence_cls):
    seq = sequence_cls([1, 2, 3, 4, 5])
    seq.block_table, seq.num_scheduled_tokens = [0, 1], 3
    sampler, model = ArgmaxSampler(), ToyModel()
    runner = ModelRunner(model, None, sampler, block_size=4)
    assert runner.run([seq], True) == [4]
    assert seq.token_ids == [1, 2, 3, 4, 5] and seq.num_cached_tokens == 0
    assert sampler.calls[0][0].shape == (1, 16)


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
def test_exception_propagates(sequence_cls):
    class BrokenModel(ToyModel):
        def forward(self, ids, positions, metadata=None):
            raise RuntimeError("execution failed")

    seq = sequence_cls([1])
    seq.num_scheduled_tokens, seq.block_table = 1, [0]
    with pytest.raises(RuntimeError, match="execution failed"):
        ModelRunner(BrokenModel(), None, ArgmaxSampler(), block_size=4).run(
            [seq], True
        )


def test_debug_validation_accepts_consistent_prefill(sequence_cls):
    seq = sequence_cls([1, 2, 3, 4])
    seq.num_scheduled_tokens, seq.block_table = 4, [0]
    runner = ModelRunner(None, None, None, block_size=4, debug_mode=True)
    ids, positions, _ = runner.prepare_prefill([seq])
    assert ids.tolist() == [1, 2, 3, 4]
    assert positions.tolist() == [0, 1, 2, 3]


def test_debug_validation_accepts_consistent_decode(sequence_cls):
    seq = sequence_cls([1, 2, 3, 4, 5])
    seq.num_cached_tokens = 4
    seq.num_scheduled_tokens = 1
    seq.block_table = [0, 1]
    runner = ModelRunner(None, None, None, block_size=4, debug_mode=True)
    ids, positions, _ = runner.prepare_decode([seq])
    assert ids.tolist() == [5]
    assert positions.tolist() == [4]


def test_debug_validation_rejects_duplicate_slots(sequence_cls):
    a, b = sequence_cls([1, 2]), sequence_cls([3, 4])
    for seq in (a, b):
        seq.num_scheduled_tokens, seq.block_table = 2, [0]
    runner = ModelRunner(None, None, None, block_size=4, debug_mode=True)
    with pytest.raises(ValueError, match="duplicate"):
        runner.prepare_prefill([a, b])


def test_debug_mode_off_skips_runner_validation(sequence_cls):
    a, b = sequence_cls([1, 2]), sequence_cls([3, 4])
    for seq in (a, b):
        seq.num_scheduled_tokens, seq.block_table = 2, [0]
    runner = ModelRunner(None, None, None, block_size=4, debug_mode=False)
    ids, _, _ = runner.prepare_prefill([a, b])
    assert ids.tolist() == [1, 2, 3, 4]


def test_debug_validation_rejects_out_of_range_slot(sequence_cls):
    seq = sequence_cls([1, 2, 3, 4, 5])
    seq.num_cached_tokens = 1
    seq.num_scheduled_tokens = 3
    seq.block_table = [99]

    class Cache:
        num_blocks = 2

    runner = ModelRunner(None, None, None, block_size=4, debug_mode=True)
    runner.kv_cache = Cache()
    with pytest.raises(ValueError, match="KV capacity|block"):
        runner.prepare_prefill([seq])


def test_runner_stores_safety_flags():
    runner = ModelRunner(
        ToyModel(), None, None, block_size=4, debug_mode=True, use_compile=False
    )
    assert runner.debug_mode is True
    assert runner.use_compile is False


def test_compile_is_decode_only():
    runner = ModelRunner(ToyModel(), None, None, block_size=4, use_compile=True)
    assert runner._forward_for(True) is runner._forward
    assert runner._forward_for(False) is runner._compiled_forward


def test_prepare_returns_attention_metadata(sequence_cls):
    seq = sequence_cls([1, 2])
    seq.num_scheduled_tokens, seq.block_table = 2, [0]
    runner = ModelRunner(None, None, None, block_size=4)
    _, _, metadata = runner.prepare_prefill([seq])
    assert metadata.is_prefill and metadata.direct_prefill
    assert metadata.cu_q_cpu == (0, 2)
    assert metadata.q_lens_cpu == (2,)
    assert metadata.slot_mapping.tolist() == [0, 1]
