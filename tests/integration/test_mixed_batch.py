import pytest
import torch

from microinfer.config import QwenConfig, SampleParameter
from microinfer.config.engine import EngineConfig
from microinfer.llm import LLM
from microinfer.model.qwen3 import QwenModel

pytestmark = pytest.mark.integration


class Tokenizer:
    eos_token_id = None

    def decode(self, ids, **kwargs):
        return str(ids)


def _model():
    torch.manual_seed(0)
    return QwenModel(
        QwenConfig(
            vocab_size=16,
            hidden_dim=8,
            MLP_inter_dim=16,
            num_q_head=2,
            num_kv_head=1,
            head_dim=4,
            num_hidden_layer=1,
            max_pos_embed=32,
        )
    )


def _llm(model):
    return LLM.from_model(
        model,
        Tokenizer(),
        block_size=2,
        num_blocks=16,
        engine_config=EngineConfig(
            max_model_len=16,
            max_num_batched_tokens=8,
            max_num_seqs=4,
        ),
    )


def test_decode_and_prefill_share_one_step_without_changing_tokens():
    reference_model = _model()
    mixed_model = _model()
    mixed_model.load_state_dict(reference_model.state_dict())
    params = SampleParameter(max_tokens=3, greedy=True, ignore_eos=True)

    reference = _llm(reference_model)
    expected = reference.generate([[1, 2], [3, 4, 5]], [params, params], use_tqdm=False)

    mixed = _llm(mixed_model)
    first = mixed.input_processor.process([1, 2], params)
    mixed.engine.add_request(first)
    mixed.engine.step()
    assert first.num_completion_tokens == 1

    second = mixed.input_processor.process([3, 4, 5], params)
    mixed.engine.add_request(second)
    before = mixed.engine.scheduler.num_mixed_steps
    mixed.engine.step()
    assert mixed.engine.scheduler.num_mixed_steps == before + 1
    assert first.num_completion_tokens == 2
    assert second.num_completion_tokens == 1

    while not mixed.engine.is_finished():
        mixed.engine.step()

    assert first.completion_token_ids == expected[0]["token_ids"]
    assert second.completion_token_ids == expected[1]["token_ids"]
    assert not mixed.engine.scheduler.block_manager.used_block_ids
