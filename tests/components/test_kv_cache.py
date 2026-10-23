import pytest
import torch

from microinfer.worker.kv_cache import KVCache
from tests.support.kv import gather

pytestmark = pytest.mark.component


def test_layout_byte_budget_and_noncontiguous_roundtrip():
    cache = KVCache(2, 3, 4, 2, 3)
    assert cache.keys.shape == cache.values.shape == (2, 3, 4, 2, 3)
    assert cache.num_bytes == 2 * 2 * 3 * 4 * 2 * 3 * 4
    keys = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    values = keys + 100
    cache.keys.fill_(-5)
    cache.values.fill_(-6)
    cache.layer(1).write_checked(torch.tensor([8, 9, 10, 11, 0]), keys, values)
    k, v = gather(cache.layer(1), [2, 0], 5)
    torch.testing.assert_close(k, keys)
    torch.testing.assert_close(v, values)
    assert (cache.keys[0] == -5).all()  # Other layer untouched.
    assert (cache.keys[1, 1] == -5).all()  # Other physical block untouched.


def test_padding_slot_does_not_write_last_physical_slot():
    cache = KVCache(1, 2, 4, 1, 2)
    cache.keys.fill_(9)
    cache.values.fill_(9)
    cache.layer(0).write_checked(torch.tensor([-1, 0]), torch.zeros(2, 1, 2), torch.ones(2, 1, 2))
    assert (cache.keys[0, -1, -1] == 9).all()
    assert (cache.keys[0, 0, 0] == 0).all()


@pytest.mark.parametrize("slots", [[0, 8], [0, -2], [0, 0]])
def test_invalid_writes_are_atomic(slots):
    cache = KVCache(1, 2, 4, 1, 2)
    cache.keys.fill_(7)
    cache.values.fill_(7)
    with pytest.raises(ValueError):
        cache.layer(0).write_checked(torch.tensor(slots), torch.zeros(2, 1, 2), torch.zeros(2, 1, 2))
    assert (cache.keys == 7).all() and (cache.values == 7).all()


@pytest.mark.parametrize("table,length", [([], 1), ([2], 1), ([0], 5), ([0], -1)])
def test_invalid_gather_rejected(table, length):
    cache = KVCache(1, 2, 4, 1, 2)
    with pytest.raises(ValueError):
        gather(cache.layer(0), table, length)


def test_fast_write_matches_checked_write():
    checked = KVCache(1, 2, 4, 1, 2)
    fast = KVCache(1, 2, 4, 1, 2)
    keys = torch.randn(3, 1, 2)
    values = torch.randn(3, 1, 2)
    slots = torch.tensor([0, 1, 2])

    checked.layer(0).write_checked(slots, keys, values)
    fast.layer(0).write_fast(slots // 4, slots % 4, keys, values)

    torch.testing.assert_close(checked.keys, fast.keys)
    torch.testing.assert_close(checked.values, fast.values)


def test_storage_preserves_dtype_and_device(device):
    cache = KVCache(1, 2, 4, 1, 2, dtype=torch.float16, device=device)
    keys = torch.ones(1, 1, 2, dtype=torch.float16, device=device)
    cache.layer(0).write_checked(torch.tensor([4], device=device), keys, keys * 2)
    k, v = gather(cache.layer(0), [1], 1)
    assert k.device == device and k.dtype == torch.float16
    torch.testing.assert_close(v, keys * 2)
