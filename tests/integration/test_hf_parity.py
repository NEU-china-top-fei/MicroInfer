"""Offline oracle: independent Transformers Qwen3, small random real checkpoints."""

import pytest
import torch

from microinfer.config import QwenConfig, SampleParameter
from microinfer.config.engine import EngineConfig
from microinfer.engine.sequence import Sequence
from microinfer.llm import LLM
from microinfer.model.loader import load_model
from microinfer.model.qwen3 import QwenModel
from microinfer.runner import ModelRunner

pytestmark = pytest.mark.integration


@pytest.fixture
def pair(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(13)
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    reference = transformers.Qwen3ForCausalLM(config).eval()
    reference.save_pretrained(tmp_path)
    model = QwenModel(QwenConfig.from_hf(config.to_dict())).eval()
    load_model(model, tmp_path)
    model.fuse_weights()
    return model, reference


@torch.inference_mode()
def test_dense_chunked_decode_and_prefix_reuse_match_transformers(pair):
    model, reference = pair
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    expected = reference(tokens, use_cache=False).logits
    actual = model.compute_logits(model(tokens, torch.arange(6)))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert model.out_logits.weight is model.embedding.weight
    runner = ModelRunner(model, None, None, block_size=2)
    cache = runner.allocate_kv_cache(4)
    seq = Sequence(tokens[0].tolist())
    seq.block_size, seq.block_table = 2, [2, 0, 3]
    for start, end, prefill in [(0, 3, True), (3, 5, True), (5, 6, False)]:
        seq.num_cached_tokens = start
        seq.num_scheduled_tokens = end - start
        if prefill:
            ids, positions, metadata = runner.prepare_prefill([seq])
        else:
            ids, positions, metadata = runner.prepare_decode([seq])
        result = runner.run_model(ids, positions, prefill, metadata)
        torch.testing.assert_close(
            result[0], expected[0, end - 1], atol=1e-5, rtol=1e-4
        )
    prefix = cache.keys[:, 2].clone()
    other = Sequence([1, 2, 7])
    other.block_table, other.num_cached_tokens, other.num_scheduled_tokens = (
        [2, 1],
        2,
        1,
    )
    ids, positions, metadata = runner.prepare_prefill([other])
    result = runner.run_model(ids, positions, True, metadata)
    oracle = reference(torch.tensor([[1, 2, 7]]), use_cache=False).logits
    torch.testing.assert_close(result[0], oracle[0, -1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(prefix, cache.keys[:, 2])


class Tokenizer:
    eos_token_id = None

    def decode(self, ids, *, skip_special_tokens):
        return " ".join(map(str, ids))


@torch.inference_mode()
def test_real_engine_generation_matches_greedy_reference_and_releases_cache(pair):
    model, reference = pair
    llm = LLM.from_model(
        model,
        Tokenizer(),
        block_size=2,
        num_blocks=8,
        engine_config=EngineConfig(max_model_len=16, max_num_batched_tokens=3),
    )
    prompts = [[1, 2, 3, 4, 5], [1, 2, 7]]
    expected = []
    for prompt in prompts:
        ids = torch.tensor([prompt])
        for _ in range(4):
            token = (
                reference(ids, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            )
            ids = torch.cat((ids, token), dim=-1)
        expected.append(ids[0, len(prompt) :].tolist())
    for _ in range(2):  # second generate exercises prefix reuse and engine reuse
        outputs = llm.generate(
            prompts, SampleParameter(max_tokens=4, greedy=True), use_tqdm=False
        )
        assert [out["token_ids"] for out in outputs] == expected
        assert not llm.engine.scheduler.block_manager.used_block_ids
        assert llm.engine.is_finished()
