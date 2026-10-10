import pytest

from microinfer.config import SampleParameter
from microinfer.engine.contracts import RequestOutput, TextOutput
from microinfer.llm import LLM

pytestmark = pytest.mark.roadmap


@pytest.fixture
def frontend(sequence_cls):
    class Input:
        def process(self, prompt, params):
            if prompt == "invalid":
                raise ValueError("bad prompt")
            return sequence_cls([1 if prompt == "a" else 2], params)

    class Output:
        def process(self, result):
            assert result.finished
            return TextOutput(
                result.seq_id,
                result.token_ids,
                str(result.token_ids),
                True,
                result.finish_reason,
            )

    class Engine:
        def __init__(self):
            self.added = []
            self.steps = 0

        def add_request(self, seq):
            self.added.append(seq)

        def has_unfinished_requests(self):
            return bool(self.added) and self.steps < 2

        def step(self):
            self.steps += 1
            assert self.steps <= 2, "frontend did not terminate"
            if self.steps == 1:
                return [RequestOutput(self.added[0].seq_id, (7,), False)]
            return [
                RequestOutput(s.seq_id, (s.token_ids[0] + 5,), True, "length")
                for s in reversed(self.added)
            ]

    engine = Engine()
    return LLM(engine, Input(), Output()), engine


def test_outputs_follow_input_order_not_completion_order(frontend):
    llm, engine = frontend
    outputs = llm.generate(["a", "b"], SampleParameter(max_tokens=1))
    assert [o.seq_id for o in outputs] == [s.seq_id for s in engine.added]
    assert [o.token_ids for o in outputs] == [(6,), (7,)]
    assert engine.steps == 2


def test_invalid_prompt_does_not_partially_enqueue(frontend):
    llm, engine = frontend
    with pytest.raises(ValueError, match="bad prompt"):
        llm.generate(["a", "invalid"], SampleParameter())
    assert not engine.added and engine.steps == 0


def test_empty_generation_does_not_step(frontend):
    llm, engine = frontend
    assert llm.generate([], SampleParameter()) == []
    assert not engine.added and engine.steps == 0
