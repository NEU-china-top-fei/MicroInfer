import pytest

from microinfer.config import SampleParameter
from microinfer.engine.sequence import SequenceStatus


def test_sequence_owns_tokens_and_sampling_values(sequence_cls):
    tokens = [1, 2, 3]
    seq = sequence_cls(tokens, SampleParameter(0.5, 7, True))
    tokens.append(99)
    assert seq.token_ids == [1, 2, 3]
    assert seq.status == SequenceStatus.WAITING
    assert not seq.is_finished
    assert (seq.temperature, seq.max_tokens, seq.ignore_eos) == (0.5, 7, True)
    assert seq.seq_id != sequence_cls([1]).seq_id


@pytest.mark.parametrize("length", [1, 3, 4, 5, 8, 9])
def test_block_partition_at_boundaries(sequence_cls, length):
    seq = sequence_cls(list(range(length)))
    chunks = [seq.block(i) for i in range(seq.num_blocks)]
    assert [token for chunk in chunks for token in chunk] == seq.token_ids
    assert all(len(chunk) == 4 for chunk in chunks[:-1])
    assert len(chunks[-1]) == seq.last_block_num_tokens
    assert seq.num_blocks == (length + 3) // 4
    for invalid in [-1, seq.num_blocks]:
        with pytest.raises(AssertionError):
            seq.block(invalid)


def test_append_preserves_prompt_and_updates_completion(sequence_cls):
    seq = sequence_cls([1, 2, 3, 4])
    seq.append_token(5)
    seq.append_token(6)
    assert len(seq) == 6
    assert seq.last_token == seq[-1] == 6
    assert seq.prompt_token_ids == [1, 2, 3, 4]
    assert seq.completion_token_ids == [5, 6]
    assert seq.num_completion_tokens == 2
    seq.status = SequenceStatus.FINISHED
    assert seq.is_finished
