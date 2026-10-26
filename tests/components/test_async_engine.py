import asyncio

import pytest

from microinfer.config import QwenConfig, SampleParameter
from microinfer.config.engine import EngineConfig
from microinfer.engine.async_engine import AsyncEngine, BackpressureError
from microinfer.engine.detokenizer import IncrementalDetokenizer
from microinfer.llm import LLM
from microinfer.model.qwen3 import QwenModel


class Tokenizer:
    eos_token_id = None

    def decode(self, ids, **kwargs):
        return "".join(str(i % 10) for i in ids)


def _llm():
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
        num_blocks=16,
        engine_config=EngineConfig(
            max_model_len=16,
            max_num_batched_tokens=8,
            max_num_seqs=4,
        ),
    )


def test_incremental_detokenizer_holds_incomplete_utf8():
    class Pieces:
        def decode(self, ids, **kwargs):
            return {
                (1,): "�",
                (1, 2): "你",
                (1, 2, 3): "你好",
            }.get(tuple(ids), "")

    detokenizer = IncrementalDetokenizer(Pieces())
    assert detokenizer.update([1]) == ""
    assert detokenizer.update([1, 2]) == "你"
    assert detokenizer.update([1, 2, 3]) == "好"
    assert detokenizer.update([1, 2, 3], finished=True) == ""


def test_async_engine_streams_and_aborts():
    llm = _llm()

    async def scenario():
        engine = AsyncEngine(llm, max_waiting=2)
        request = await engine.add_request(
            [1, 2], SampleParameter(max_tokens=4, greedy=True, ignore_eos=True)
        )
        events = [event async for event in engine.stream(request)]
        assert events[-1].token_ids and len(events[-1].token_ids) == 4
        assert events[-1].finished and events[-1].finish_reason == "length"
        assert events[-1].token_ids == request.token_ids

        long = await engine.add_request(
            [3], SampleParameter(max_tokens=100, greedy=True, ignore_eos=True)
        )
        first = await anext(engine.stream(long))
        assert not first.finished
        assert await engine.abort(long.seq_id)
        terminal = await engine.collect(long)
        assert terminal.finished and terminal.finish_reason == "abort"
        await engine.stop()
        assert not engine.engine.scheduler.block_manager.used_block_ids

    asyncio.run(scenario())


def test_async_engine_backpressure_and_release():
    llm = _llm()

    async def scenario():
        engine = AsyncEngine(llm, max_waiting=1)
        request = await engine.add_request(
            [1], SampleParameter(max_tokens=2, greedy=True, ignore_eos=True)
        )
        with pytest.raises(BackpressureError):
            await engine.add_request(
                [2], SampleParameter(max_tokens=2, greedy=True, ignore_eos=True)
            )
        await engine.collect(request)
        # collect releases finished bookkeeping; a new request is admitted.
        second = await engine.add_request(
            [2], SampleParameter(max_tokens=1, greedy=True, ignore_eos=True)
        )
        await engine.collect(second)
        await engine.stop()

    asyncio.run(scenario())
