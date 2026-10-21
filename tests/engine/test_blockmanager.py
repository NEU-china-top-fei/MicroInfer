from collections import Counter

import pytest

from microinfer.engine.blockmanager import Block, BlockManager


def assert_ownership(manager, *sequences):
    """Check physical ownership against actual sequence references."""
    refs = Counter(bid for seq in sequences for bid in seq.block_table)
    free = list(manager.free_block_ids)
    used = manager.used_block_ids
    assert len(free) == len(set(free))
    assert set(free).isdisjoint(used)
    assert set(free) | used == set(range(len(manager.blocks)))
    assert used == set(refs)
    for block in manager.blocks:
        assert block.ref_count == refs[block.block_id]


def seed_completed_blocks(manager, seq, count):
    """Arrange valid cache metadata to test allocation independently of hash_blocks.

    No KV tensors are involved in this CPU metadata component. Integration tests
    below use the real hash_blocks instead of this setup helper.
    """
    prefix = -1
    for i in range(count):
        tokens = seq.block(i)
        assert len(tokens) == manager.block_size
        prefix = manager.compute_hash(tokens, prefix)
        block = manager.blocks[seq.block_table[i]]
        block.update(prefix, tokens)
        manager.hash_to_block_id[prefix] = block.block_id


def test_block_reset_discards_previous_content():
    block = Block(3)
    assert block.ref_count == 0
    block.update(123, [1, 2])
    block.reset()
    assert (block.block_id, block.ref_count, block.hash, block.token_ids) == (
        3,
        1,
        -1,
        [],
    )


def test_hash_depends_on_prefix_and_token_order():
    h = BlockManager.compute_hash
    assert h([1, 2]) == h([1, 2])
    assert h([1, 2]) != h([2, 1])
    assert h([1, 2], h([3, 4])) != h([1, 2], h([4, 3]))


@pytest.mark.parametrize("length", [1, 4, 5, 8, 9])
def test_allocate_release_preserves_ownership(sequence_cls, length):
    manager = BlockManager(3, 4)
    seq = sequence_cls(list(range(length)))
    assert manager.can_allocate(seq) == 0
    manager.allocate(seq, 0)
    assert len(seq.block_table) == seq.num_blocks
    assert not manager.hash_to_block_id  # Allocation does not publish usable KV.
    assert_ownership(manager, seq)
    manager.deallocate(seq)
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert_ownership(manager)


def test_capacity_check_is_read_only(sequence_cls):
    manager = BlockManager(1, 4)
    assert manager.can_allocate(sequence_cls([1, 2, 3, 4, 5])) == -1
    assert_ownership(manager)


def test_shared_prefix_freed_only_after_last_owner(sequence_cls):
    manager = BlockManager(3, 4)
    a, b = sequence_cls([1, 2, 3, 4, 5]), sequence_cls([1, 2, 3, 4, 6])
    manager.allocate(a, 0)
    seed_completed_blocks(manager, a, 1)
    assert manager.can_allocate(b) == 1
    manager.allocate(b, 1)
    shared = a.block_table[0]
    assert b.block_table[0] == shared
    assert b.num_cached_tokens == 4
    assert_ownership(manager, a, b)
    manager.deallocate(a)
    assert shared not in manager.free_block_ids
    assert_ownership(manager, b)
    manager.deallocate(b)
    assert_ownership(manager)


def test_reactivate_free_cached_prefix(sequence_cls):
    manager = BlockManager(2, 4)
    a = sequence_cls([1, 2, 3, 4, 5])
    manager.allocate(a, 0)
    seed_completed_blocks(manager, a, 1)
    cached_id = a.block_table[0]
    manager.deallocate(a)
    b = sequence_cls([1, 2, 3, 4, 6])
    assert manager.can_allocate(b) == 1
    manager.allocate(b, 1)
    assert b.block_table[0] == cached_id
    assert_ownership(manager, b)


@pytest.mark.parametrize("length", [3, 4, 8])
def test_last_block_is_never_reused(sequence_cls, length):
    manager = BlockManager(4, 4)
    a = sequence_cls(list(range(length)))
    manager.allocate(a, 0)
    seed_completed_blocks(manager, a, length // 4)
    b = sequence_cls(list(range(length)))
    assert manager.can_allocate(b) == a.num_blocks - 1


@pytest.mark.parametrize(
    "length, needs_block", [(3, False), (4, True), (7, False), (8, True)]
)
def test_decode_append_at_boundary(sequence_cls, length, needs_block):
    manager = BlockManager(4, 4)
    seq = sequence_cls(list(range(length)))
    manager.allocate(seq, 0)
    previous = list(seq.block_table)
    seq.append_token(99)
    assert manager.can_append(seq)
    manager.may_append(seq)
    assert len(seq.block_table) == len(previous) + needs_block
    assert seq.block_table[: len(previous)] == previous
    assert_ownership(manager, seq)
    manager.deallocate(seq)
    assert_ownership(manager)


def test_decode_capacity_and_finish_before_next_allocation(sequence_cls):
    manager = BlockManager(1, 4)
    seq = sequence_cls([1, 2, 3, 4])
    manager.allocate(seq, 0)
    seq.append_token(5)
    assert not manager.can_append(seq)
    manager.deallocate(seq)  # EOS may arrive before the new token gets a KV slot.
    assert_ownership(manager)


def test_free_cache_hit_does_not_create_extra_capacity(sequence_cls):
    manager = BlockManager(2, 4)
    cached = sequence_cls([1, 2, 3, 4])
    manager.allocate(cached, 0)
    seed_completed_blocks(manager, cached, 1)
    manager.deallocate(cached)
    busy = sequence_cls([9])
    manager.allocate(busy, 0)
    assert manager.can_allocate(sequence_cls([1, 2, 3, 4, 5])) == -1
    assert_ownership(manager, busy)


def test_eviction_removes_stale_hash(sequence_cls):
    manager = BlockManager(1, 4)
    a = sequence_cls([1, 2, 3, 4])
    manager.allocate(a, 0)
    seed_completed_blocks(manager, a, 1)
    old_hash = manager.compute_hash(a.token_ids)
    manager.deallocate(a)
    b = sequence_cls([8, 9, 10, 11])
    manager.allocate(b, 0)
    assert old_hash not in manager.hash_to_block_id
    assert_ownership(manager, b)


def test_eviction_does_not_delete_mapping_owned_by_another_block(sequence_cls):
    manager = BlockManager(2, 4)
    a, b = sequence_cls([1, 2, 3, 4]), sequence_cls([1, 2, 3, 4])
    for seq in (a, b):
        manager.allocate(seq, 0)
        seed_completed_blocks(manager, seq, 1)
    manager.deallocate(a)
    manager.allocate(sequence_cls([8]), 0)
    assert (
        manager.hash_to_block_id[manager.compute_hash(b.token_ids)] == b.block_table[0]
    )


def test_hash_hit_checks_tokens(sequence_cls):
    manager = BlockManager(3, 4)
    a = sequence_cls([9, 9, 9, 9])
    manager.allocate(a, 0)
    # Deliberately arrange a collision without relying on a real 64-bit collision.
    manager.hash_to_block_id[manager.compute_hash([1, 2, 3, 4])] = a.block_table[0]
    assert manager.can_allocate(sequence_cls([1, 2, 3, 4, 5])) == 0


@pytest.mark.parametrize("scheduled", [0, 3, 4, 5, 8, 9])
def test_hash_publishes_only_completed_full_blocks(sequence_cls, scheduled):
    manager = BlockManager(3, 4)
    seq = sequence_cls(list(range(9)))
    manager.allocate(seq, 0)
    seq.num_scheduled_tokens = scheduled
    manager.hash_blocks(seq)
    expected = {}
    prefix = -1
    for i in range(scheduled // 4):
        prefix = manager.compute_hash(seq.block(i), prefix)
        expected[prefix] = seq.block_table[i]
    assert manager.hash_to_block_id == expected
    for key, bid in expected.items():
        assert manager.blocks[bid].hash == key


@pytest.mark.integration
def test_chunked_prefill_then_prefix_reuse(sequence_cls):
    manager = BlockManager(5, 4)
    seq = sequence_cls(list(range(9)))
    manager.allocate(seq, 0)
    for chunk in [3, 2, 4]:
        seq.num_scheduled_tokens = chunk
        manager.hash_blocks(seq)
        seq.num_cached_tokens += chunk
        seq.num_scheduled_tokens = 0
    second = sequence_cls(list(range(8)) + [99])
    assert manager.can_allocate(second) == 2
    manager.allocate(second, 2)
    assert second.block_table[:2] == seq.block_table[:2]
    assert_ownership(manager, seq, second)


def test_decode_completes_and_publishes_block(sequence_cls):
    manager = BlockManager(2, 4)
    seq = sequence_cls([1, 2, 3])
    manager.allocate(seq, 0)
    seq.num_cached_tokens = 3
    seq.append_token(4)
    manager.may_append(seq)
    seq.num_scheduled_tokens = 1
    manager.hash_blocks(seq)
    bid = seq.block_table[0]
    key = manager.compute_hash([1, 2, 3, 4])
    assert manager.hash_to_block_id == {key: bid}
    assert manager.blocks[bid].token_ids == [1, 2, 3, 4]
    assert manager.blocks[bid].hash == key
