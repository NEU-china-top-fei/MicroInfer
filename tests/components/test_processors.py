import pytest

from microinfer.config import SampleParameter
from microinfer.engine.contracts import RequestOutput
from microinfer.engine.processor import InputProcessor, OutputProcessor

pytestmark = [pytest.mark.component, pytest.mark.extension]


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is True
        return {"hello": [1, 2], "empty": []}[text]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return {(4,): "�", (4, 5): "你好", (4, 5, 0): "你好"}[tuple(ids)]


def test_text_input_preserves_sampling_parameters():
    seq = InputProcessor(Tokenizer(), max_model_len=8).process(
        "hello", SampleParameter(0.7, 3, True)
    )
    assert seq.token_ids == [1, 2]
    assert (seq.temperature, seq.max_tokens, seq.ignore_eos) == (0.7, 3, True)


def test_token_input_is_copied_without_tokenizing():
    tokens = [1, 2]
    seq = InputProcessor(object(), max_model_len=8).process(tokens, SampleParameter())
    tokens.append(3)
    assert seq.token_ids == [1, 2]


@pytest.mark.parametrize("prompt", [[], "empty", [-1], [True], [1.5], [1] * 8])
def test_invalid_input_rejected_before_sequence_creation(prompt):
    with pytest.raises(ValueError):
        InputProcessor(Tokenizer(), max_model_len=8).process(prompt, SampleParameter())


def test_decode_reprocesses_whole_completion_for_multitoken_characters():
    processor = OutputProcessor(Tokenizer())
    partial = processor.process(RequestOutput(3, (4,), False))
    complete = processor.process(RequestOutput(3, (4, 5, 0), True, "eos"))
    assert partial.text == "�"
    assert complete.text == "你好"
    assert complete.token_ids == (4, 5, 0)
    assert complete.finished and complete.finish_reason == "eos"
    assert complete.seq_id == 3
