import pytest
import torch

from microinfer.engine.contracts import ScheduledSequence, SchedulerOutput
from microinfer.runner import ModelRunner
from microinfer.worker.batch import prepare_batch
from microinfer.worker.kv_cache import KVCache

pytestmark = [pytest.mark.component, pytest.mark.integration]


@pytest.fixture
def model_and_cache(tiny_config, seeded_torch):
    def build():
        from microinfer.model.qwen3 import QwenModel

        tiny_config.rope_base = 10000.0
        model = QwenModel(tiny_config).eval()

        def new_cache():
            return KVCache(
                tiny_config.num_hidden_layer,
                3,
                4,
                tiny_config.num_kv_head,
                tiny_config.head_dim,
            )

        return model, new_cache

    return build


def run_cached(model, batch, cache):
    # Binding is a runner/setup concern, never a model.forward argument.
    ModelRunner(model, cache, None, block_size=cache.block_size)
    return model.forward_batch(batch)


def packed(tokens, start, blocks, seq_id=1):
    return ScheduledSequence(seq_id, tuple(tokens), start, tuple(blocks), True)


@pytest.mark.parametrize("chunks", [[3, 2], [4, 1], [1, 1, 1, 1, 1]])
def test_full_prefill_equals_chunked_prefill_and_decode(model_and_cache, chunks):
    model, new_cache = model_and_cache()
    tokens = [1, 2, 3, 4, 5]
    with torch.inference_mode():
        full = run_cached(
            model,
            prepare_batch(SchedulerOutput((packed(tokens, 0, [2, 0]),)), 4),
            new_cache(),
        )
        cache, pieces, start = new_cache(), [], 0
        for count in chunks:
            item = packed(tokens[start : start + count], start, [2, 0])
            pieces.append(
                run_cached(model, prepare_batch(SchedulerOutput((item,)), 4), cache)
            )
            start += count
    assert full.shape == (5, 32)
    torch.testing.assert_close(torch.cat(pieces), full, atol=1e-5, rtol=1e-4)


def test_packed_requests_match_individual_execution(model_and_cache):
    model, new_cache = model_and_cache()
    a, b = packed([1, 2, 3], 0, [2], 1), packed([7, 8], 0, [0], 2)
    with torch.inference_mode():
        combined = run_cached(
            model, prepare_batch(SchedulerOutput((a, b)), 4), new_cache()
        )
        separate = [
            run_cached(model, prepare_batch(SchedulerOutput((item,)), 4), new_cache())
            for item in [a, b]
        ]
    torch.testing.assert_close(combined, torch.cat(separate), atol=1e-5, rtol=1e-4)


def test_shared_full_prefix_is_read_only_and_matches_recomputation(model_and_cache):
    model, new_cache = model_and_cache()
    cache = new_cache()
    with torch.inference_mode():
        run_cached(
            model,
            prepare_batch(SchedulerOutput((packed([1, 2, 3, 4, 5], 0, [2, 0]),)), 4),
            cache,
        )
        prefix_keys, prefix_values = (
            cache.keys[:, 2].clone(),
            cache.values[:, 2].clone(),
        )
        reuse = run_cached(
            model,
            prepare_batch(SchedulerOutput((packed([6], 4, [2, 1], 2),)), 4),
            cache,
        )
        fresh = run_cached(
            model,
            prepare_batch(SchedulerOutput((packed([1, 2, 3, 4, 6], 0, [2, 1], 2),)), 4),
            new_cache(),
        )
    torch.testing.assert_close(reuse[0], fresh[-1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(cache.keys[:, 2], prefix_keys)
    torch.testing.assert_close(cache.values[:, 2], prefix_values)
