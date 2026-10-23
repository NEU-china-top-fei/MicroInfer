"""Cache ownership, layer binding, and AttentionMetadata adaptation."""

import pytest
import torch

from microinfer.layer.cached_attention import CachedAttention
from microinfer.layer.metadata import AttentionMetadata
from microinfer.runner import ModelRunner
from microinfer.worker.kv_cache import KVCache
from tests.support.kv import gather


def test_layer_view_shares_storage_and_cannot_write_other_layers():
    pool = KVCache(2, 3, 2, 1, 4)
    view = pool.layer(1)
    assert (
        view.keys.untyped_storage().data_ptr() == pool.keys.untyped_storage().data_ptr()
    )
    assert (
        view.values.untyped_storage().data_ptr()
        == pool.values.untyped_storage().data_ptr()
    )
    view.write_checked(
        torch.tensor([4, 1]), torch.ones(2, 1, 4), torch.full((2, 1, 4), 3.0)
    )
    assert pool.keys[1, 2, 0].eq(1).all()
    assert pool.values[1, 0, 1].eq(3).all()
    assert pool.keys[0].eq(0).all() and pool.values[0].eq(0).all()


@pytest.mark.parametrize("slots", [[], [-1, -1]])
def test_empty_and_padding_only_writes_leave_storage_unchanged(slots):
    view = KVCache(1, 1, 2, 1, 4).layer(0)
    values = torch.ones(len(slots), 1, 4)
    view.write_checked(torch.tensor(slots, dtype=torch.long), values, values)
    assert view.keys.eq(0).all() and view.values.eq(0).all()


def test_layer_write_validates_both_inputs_before_mutation():
    view = KVCache(1, 1, 2, 1, 4).layer(0)
    with pytest.raises(ValueError):
        view.write_checked(torch.tensor([0]), torch.ones(1, 1, 4), torch.ones(2, 1, 4))
    assert view.keys.eq(0).all()


@pytest.fixture
def model(tiny_config):
    from microinfer.model.qwen3 import QwenModel

    tiny_config.rope_base = 10000.0
    return QwenModel(tiny_config)


def test_runner_allocates_and_rebinds_every_layer_without_model_owning_pool(model):
    assert not hasattr(model, "kv_cache")
    runner = ModelRunner(model, None, None, block_size=4)
    first = runner.allocate_kv_cache(3)
    state_keys = set(model.state_dict())
    second = runner.allocate_kv_cache(2)
    assert runner.kv_cache is second
    for i, layer in enumerate(model.transformer_layers):
        view = layer.attention.attn.cache
        assert view.keys.data_ptr() == second.keys[i].data_ptr()
        view.write_checked(torch.tensor([0]), torch.ones(1, 2, 4), torch.ones(1, 2, 4))
    assert first.keys.eq(0).all()
    assert second.keys[:, 0, 0].eq(1).all()
    assert set(model.state_dict()) == state_keys
    assert not any("cache" in key for key in state_keys)


@pytest.mark.parametrize("mismatch", ["layers", "heads", "blocks", "dtype"])
def test_invalid_rebind_preserves_existing_bindings(model, mismatch):
    runner = ModelRunner(model, None, None, block_size=4)
    original = runner.allocate_kv_cache(2)
    bad = KVCache(
        1 if mismatch == "layers" else 2,
        2,
        2 if mismatch == "blocks" else 4,
        1 if mismatch == "heads" else 2,
        4,
        dtype=torch.float16 if mismatch == "dtype" else torch.float32,
    )
    with pytest.raises(ValueError):
        runner.bind_kv_cache(bad)
    assert runner.kv_cache is original
    for i, layer in enumerate(model.transformer_layers):
        assert layer.attention.attn.cache.keys.data_ptr() == original.keys[i].data_ptr()


def dense_reference(q, k, v):
    result = torch.empty_like(q)
    repeats = q.shape[1] // k.shape[1]
    for i in range(len(q)):
        for h in range(q.shape[1]):
            weights = (k[: i + 1, h // repeats] @ q[i, h] / q.shape[-1] ** 0.5).softmax(
                0
            )
            result[i, h] = weights @ v[: i + 1, h // repeats]
    return result


def test_cached_chunks_and_decode_match_full_prefill_with_noncontiguous_blocks(
    seeded_torch,
):
    q, k, v = torch.randn(6, 4, 4), torch.randn(6, 2, 4), torch.randn(6, 2, 4)
    expected = dense_reference(q, k, v)
    backend = CachedAttention()
    pool = KVCache(2, 4, 2, 2, 4)
    backend.bind_cache(pool.layer(1))
    # Logical blocks map to physical blocks 2,0,3. Last chunk is decode.
    slots = torch.tensor([4, 5, 0, 1, 6, 7])
    outputs = []
    for start, end in [(0, 3), (3, 5), (5, 6)]:
        prefill = start != 5
        if prefill:
            metadata = AttentionMetadata.build(
                is_prefill=True,
                cu_seqlens_q=torch.tensor([0, end - start]),
                cu_seqlens_k=torch.tensor([0, end]),
                max_seqlen_q=end - start,
                max_seqlen_k=end,
                slot_mapping=slots[start:end],
                block_tables=None if start == 0 else torch.tensor([[2, 0, 3]]),
            )
        else:
            metadata = AttentionMetadata.build(
                is_prefill=False,
                slot_mapping=slots[start:end],
                context_lens=torch.tensor([end]),
                block_tables=torch.tensor([[2, 0, 3]]),
            )
        outputs.append(backend(q[start:end], k[start:end], v[start:end], metadata))
    torch.testing.assert_close(torch.cat(outputs), expected, atol=1e-5, rtol=1e-5)
    assert pool.keys[0].eq(0).all()
    torch.testing.assert_close(gather(pool.layer(1), [2, 0, 3], 6)[0], k)


def test_no_prefix_prefill_isolates_requests_without_cache(seeded_torch):
    q, k, v = torch.randn(5, 4, 4), torch.randn(5, 2, 4), torch.randn(5, 2, 4)
    metadata = AttentionMetadata.build(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 3, 5]),
        cu_seqlens_k=torch.tensor([0, 3, 5]),
        max_seqlen_q=3,
        max_seqlen_k=5,
    )
    actual = CachedAttention()(q, k, v, metadata)
    expected = torch.cat(
        [dense_reference(q[:3], k[:3], v[:3]), dense_reference(q[3:], k[3:], v[3:])]
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_fast_cache_write_matches_checked_path(seeded_torch):
    q, k, v = torch.randn(3, 4, 4), torch.randn(3, 2, 4), torch.randn(3, 2, 4)
    pool = KVCache(1, 2, 4, 2, 4)
    backend = CachedAttention()
    backend.bind_cache(pool.layer(0))
    metadata = AttentionMetadata.build(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 3]),
        cu_seqlens_k=torch.tensor([0, 3]),
        max_seqlen_q=3,
        max_seqlen_k=3,
        slot_mapping=torch.tensor([0, 1, 2]),
        validate_cache_write=False,
    )
    actual = backend(q, k, v, metadata)
    expected = dense_reference(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pool.layer(0).keys[0, :3], k)
    torch.testing.assert_close(pool.layer(0).values[0, :3], v)


def test_decode_without_bound_cache_fails_clearly():
    metadata = AttentionMetadata.build(
        is_prefill=False,
        context_lens=torch.tensor([1]),
        block_tables=torch.tensor([[0]]),
    )
    q = torch.zeros(1, 1, 4)
    with pytest.raises(RuntimeError, match="bound layer cache"):
        CachedAttention()(q, q, q, metadata)
