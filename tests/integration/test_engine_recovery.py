import pytest

from microinfer.config import QwenConfig, SampleParameter
from microinfer.config.engine import EngineConfig
from microinfer.engine.blockmanager import BlockManager
from microinfer.engine.scheduler import Scheduler
from microinfer.engine.sequence import Sequence
from microinfer.llm import LLM
from microinfer.model.qwen3 import QwenModel

pytestmark = pytest.mark.integration


class Tokenizer:
    eos_token_id = None

    def decode(self, ids, **kwargs):
        return str(ids)


@pytest.fixture
def llm():
    model = QwenModel(
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
    return LLM.from_model(
        model,
        Tokenizer(),
        block_size=2,
        num_blocks=4,
        engine_config=EngineConfig(max_model_len=16, max_num_batched_tokens=2),
    )


def test_failed_runner_releases_all_admitted_requests_and_can_retry(llm, monkeypatch):
    runner = llm.engine.model_runner
    actual = runner.run_model

    def fail_after_write(*args):
        actual(*args)
        raise RuntimeError("injected after KV writes")

    monkeypatch.setattr(runner, "run_model", fail_after_write)
    with pytest.raises(RuntimeError, match="injected"):
        llm.generate([[1, 2, 3], [4]], SampleParameter(max_tokens=2), use_tqdm=False)
    assert llm.engine.is_finished()
    assert not llm.engine.scheduler.block_manager.used_block_ids
    monkeypatch.setattr(runner, "run_model", actual)
    assert (
        len(
            llm.generate([[1, 2, 3]], SampleParameter(max_tokens=2), use_tqdm=False)[0][
                "token_ids"
            ]
        )
        == 2
    )


def test_generation_stops_at_cache_context_capacity(llm):
    output = llm.generate(
        [[1, 2, 3]], SampleParameter(max_tokens=100, greedy=True), use_tqdm=False
    )
    assert len(output[0]["token_ids"]) == 5  # Pool holds 8 tokens including prompt.
    assert not llm.engine.scheduler.block_manager.used_block_ids


def test_failed_admission_rolls_back_previous_requests(llm):
    with pytest.raises(ValueError):
        llm.generate([[1], [1] * 8], use_tqdm=False)
    assert llm.engine.is_finished()
    with pytest.raises(ValueError, match="one sampling"):
        llm.generate([[1]], [], use_tqdm=False)


def test_abort_during_chunked_prefill_and_empty_step(llm):
    seq = Sequence([1, 2, 3, 4, 5])
    llm.engine.add_request(seq)
    llm.engine.step()
    assert seq.num_cached_tokens == 2
    assert llm.engine.abort(seq.seq_id)
    assert not llm.engine.abort(seq.seq_id)
    assert not llm.engine.scheduler.block_manager.used_block_ids
    assert llm.engine.step() == ([], 0)


def test_block_size_one_and_inflight_guard():
    scheduler = Scheduler(EngineConfig(max_model_len=8), BlockManager(8, 1))
    seq = Sequence([1, 2], SampleParameter(max_tokens=3))
    scheduler.add(seq)
    for token in [3, 4, 5]:
        seqs, mode = scheduler.schedule()
        with pytest.raises(RuntimeError, match="postprocessed"):
            scheduler.schedule()
        scheduler.postprocess(seqs, [token], mode)
    assert seq.completion_token_ids == [3, 4, 5]
    assert scheduler.is_finished() and not scheduler.block_manager.used_block_ids


def test_chunked_recompute_does_not_sample_before_known_generated_tokens():
    scheduler = Scheduler(
        EngineConfig(max_model_len=32, max_num_batched_tokens=2), BlockManager(8, 4)
    )
    seq = Sequence([1, 2], SampleParameter(max_tokens=8))
    scheduler.add(seq)
    for token in [3, 4, 5]:
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [token], mode)
    scheduler.running.remove(seq)
    scheduler.preempt(seq)
    # Clear prefix lookup to force full recomputation, including generated tokens.
    scheduler.block_manager.hash_to_block_id.clear()
    for _ in range(2):
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [15], mode)
        assert seq.completion_token_ids == [3, 4, 5]
    seqs, mode = scheduler.schedule()
    scheduler.postprocess(seqs, [6], mode)
    assert seq.completion_token_ids == [3, 4, 5, 6]
