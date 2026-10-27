import torch

from microinfer.config import QwenConfig, SampleParameter
from microinfer.config.engine import EngineConfig
from microinfer.engine.blockmanager import BlockManager
from microinfer.engine.scheduler import Scheduler
from microinfer.engine.sequence import Sequence, SequenceStatus
from microinfer.engine.speculative import NgramSpeculator
from microinfer.llm import LLM
from microinfer.model.qwen3 import QwenModel


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


def _llm(model, *, speculative):
    config = EngineConfig(
        max_model_len=16,
        max_num_batched_tokens=8,
        max_num_seqs=4,
        use_speculative=speculative,
    )
    return LLM.from_model(
        model,
        Tokenizer(),
        block_size=2,
        num_blocks=16,
        engine_config=config,
    )


def test_ngram_speculator_finds_previous_continuation():
    speculator = NgramSpeculator(ngram=3, max_drafts=8)
    tokens = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5]
    assert speculator.propose(tokens) == [6, 1, 2, 3, 4, 5]
    assert speculator.propose([1, 2, 3]) == []


def test_speculative_generation_matches_greedy_baseline():
    baseline_model = _model()
    spec_model = _model()
    spec_model.load_state_dict(baseline_model.state_dict())
    params = SampleParameter(max_tokens=4, greedy=True, ignore_eos=True)

    baseline = _llm(baseline_model, speculative=False)
    expected = baseline.generate([[1, 2, 3]], params, use_tqdm=False)

    speculative = _llm(spec_model, speculative=True)

    class AlwaysDraft:
        def propose(self, seq, limit=None):
            return [1, 1]

    speculative.engine.speculator = AlwaysDraft()
    actual = speculative.generate([[1, 2, 3]], params, use_tqdm=False)

    assert actual == expected
    stats = speculative.engine.stats()
    assert stats["num_spec_steps"] > 0
    assert stats["num_spec_draft_tokens"] > 0
    assert not speculative.engine.scheduler.block_manager.used_block_ids


def test_speculative_is_skipped_for_temperature_sampling():
    llm = _llm(_model(), speculative=True)
    params = SampleParameter(
        temperature=0.8, max_tokens=2, greedy=False, ignore_eos=True
    )
    llm.generate([[1, 2, 3]], params, use_tqdm=False)
    assert llm.engine.stats()["num_spec_steps"] == 0


def test_commit_spec_keeps_last_draft_for_normal_decode():
    scheduler = Scheduler(EngineConfig(max_model_len=32), BlockManager(8, 2))
    seq = Sequence([1, 2, 3, 4], SampleParameter(max_tokens=10, greedy=True))
    seq.block_size = 2
    seq.num_cached_tokens = 3
    seq.token_ids = [1, 2, 3, 4, 5, 6]
    seq.num_tokens = 6
    seq.status = SequenceStatus.RUNNING
    seq.spec_base_len = 4
    seq.spec_draft_ids = (5, 6)
    seq.num_scheduled_tokens = 3
    seq.is_prefill = True
    scheduler.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = 3
    scheduler.running.append(seq)
    scheduler.commit_spec([seq], [2])
    assert seq.completion_token_ids == [5, 6]
    assert seq.num_cached_tokens == 5 and seq.num_scheduled_tokens == 0
    assert seq.spec_draft_ids == () and not seq.is_prefill
    assert seq in scheduler.running
    scheduler.abort(seq.seq_id)
