import pytest

from microinfer.config import SampleParameter
from microinfer.engine.contracts import TextOutput
from microinfer.llm import LLM

pytestmark = [pytest.mark.roadmap, pytest.mark.nano]


@pytest.fixture
def frontend(sequence_cls):
    class Input:
        def process(self, prompt, params):
            if prompt == "invalid":
                raise ValueError("bad prompt")
            return sequence_cls([1 if prompt == "a" else 2], params)

    class Output:
        def process(self, result):
            return TextOutput(
                result.seq_id,
                tuple(result.token_ids),
                str(list(result.token_ids)),
                True,
            )

    class Engine:
        def __init__(self):
            self.added, self.steps = [], 0

        def add_request(self, seq):
            self.added.append(seq)

        def is_finished(self):
            return not self.added or self.steps >= 2

        def step(self):
            self.steps += 1
            assert self.steps <= 2
            if self.steps == 1:
                return [], 2
            return [(s.seq_id, [s.token_ids[0] + 5]) for s in reversed(self.added)], -2

    engine = Engine()
    return LLM(engine, Input(), Output()), engine


@pytest.mark.parametrize("per_request", [False, True])
def test_generate_returns_nano_dicts_in_prompt_order(frontend, per_request):
    llm, engine = frontend
    params = (
        [SampleParameter(0.5), SampleParameter(0.8)]
        if per_request
        else SampleParameter()
    )
    outputs = llm.generate(["a", "b"], params, use_tqdm=False)
    assert outputs == [
        {"text": "[6]", "token_ids": [6]},
        {"text": "[7]", "token_ids": [7]},
    ]
    assert engine.steps == 2
    assert [s.temperature for s in engine.added] == (
        [0.5, 0.8] if per_request else [1.0, 1.0]
    )


@pytest.mark.extension
def test_invalid_prompt_does_not_partially_enqueue(frontend):
    llm, engine = frontend
    with pytest.raises(ValueError, match="bad prompt"):
        llm.generate(["a", "invalid"], SampleParameter(), use_tqdm=False)
    assert not engine.added and engine.steps == 0


def test_empty_generation_does_not_step(frontend):
    llm, engine = frontend
    assert llm.generate([], SampleParameter(), use_tqdm=False) == []
    assert not engine.added and engine.steps == 0
