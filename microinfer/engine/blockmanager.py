import heapq

import numpy as np
import xxhash

from microinfer.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.last_used = 0

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if (
            type(num_blocks) is not int
            or type(block_size) is not int
            or min(num_blocks, block_size) <= 0
        ):
            raise ValueError("num_blocks and block_size must be positive integers")
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self._free_blocks: set[int] = set(range(num_blocks))
        # Lazy LRU heap: entries are (last_used, block_id); stale entries are
        # discarded when popped. The set above is the source of truth.
        self._free_heap: list[tuple[int, int]] = [(0, i) for i in range(num_blocks)]
        heapq.heapify(self._free_heap)
        self._clock = 0
        self.block_id_to_hashes: dict[int, set[int]] = {}
        self.used_block_ids: set[int] = set()
        # Observability only; counters do not affect allocation decisions.
        self.prefix_cache_hit_tokens = 0
        self.prefix_cache_hit_blocks = 0
        self.prefix_cache_evictions = 0

    @property
    def free_block_ids(self):
        """Set view used by diagnostics/tests; mutation stays internal."""
        return self._free_blocks

    def _touch(self, block_id: int):
        self._clock += 1
        self.blocks[block_id].last_used = self._clock
        if block_id not in self._free_blocks:
            return
        heapq.heappush(self._free_heap, (self._clock, block_id))
        # Repeated prefix lookups leave stale heap entries. Rebuild when the
        # heap has grown a few times larger than the pool, keeping it bounded.
        if len(self._free_heap) > max(64, 4 * len(self.blocks)):
            self._free_heap = [(self.blocks[i].last_used, i) for i in self._free_blocks]
            heapq.heapify(self._free_heap)

    def _evict_hashes(self, block_id: int):
        hashes = self.block_id_to_hashes.pop(block_id, set())
        block = self.blocks[block_id]
        if block.hash != -1:
            hashes.add(block.hash)
        for hash_value in hashes:
            if self.hash_to_block_id.get(hash_value) == block_id:
                del self.hash_to_block_id[hash_value]

    def _publish_hash(self, hash_value: int, block_id: int):
        self.hash_to_block_id[hash_value] = block_id
        self.block_id_to_hashes.setdefault(block_id, set()).add(hash_value)

    def stats(self) -> dict:
        total = len(self.blocks)
        used = len(self.used_block_ids)
        return {
            "num_blocks": total,
            "used_blocks": used,
            "free_blocks": len(self._free_blocks),
            "kv_utilization": used / total if total else 0.0,
            "prefix_cache_entries": len(self.hash_to_block_id),
            "prefix_cache_hit_tokens": self.prefix_cache_hit_tokens,
            "prefix_cache_hit_blocks": self.prefix_cache_hit_blocks,
            "prefix_cache_evictions": self.prefix_cache_evictions,
        }

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """Hash a block, including the hash of its full prefix block."""
        hasher = xxhash.xxh64()
        if prefix != -1:
            hasher.update(prefix.to_bytes(8, "little"))
        hasher.update(np.array(token_ids).tobytes())
        return hasher.intdigest()

    def _allocate_block(self) -> int:
        while self._free_heap:
            last_used, block_id = heapq.heappop(self._free_heap)
            if (
                block_id in self._free_blocks
                and last_used == self.blocks[block_id].last_used
            ):
                break
        else:
            raise RuntimeError("KV cache has no free blocks")

        self._free_blocks.remove(block_id)
        self.used_block_ids.add(block_id)
        block = self.blocks[block_id]
        if block.token_ids:
            self.prefix_cache_evictions += 1
        self._evict_hashes(block_id)
        block.reset()
        self._touch(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        # Keep KV and hash metadata until the block is reused.
        self.used_block_ids.remove(block_id)
        self._free_blocks.add(block_id)
        self._touch(block_id)

    def reset_prefix_cache(self):
        """Drop all cached prefix metadata; only valid when the pool is idle."""
        if self.used_block_ids:
            raise RuntimeError("cannot reset prefix cache while blocks are in use")
        self.hash_to_block_id.clear()
        self.block_id_to_hashes.clear()
        for block in self.blocks:
            block.hash = -1
            block.token_ids = []
        self.prefix_cache_hit_tokens = 0
        self.prefix_cache_hit_blocks = 0
        self.prefix_cache_evictions = 0

    def can_allocate(self, seq: Sequence) -> int:
        """Return reusable block count, or -1 if the pool cannot fit the request.

        The last block is never reused so a partial block cannot expose tokens
        beyond the true sequence length. Hash collisions are checked by tokens.
        """
        num_blocks_reused = 0
        free_cache_hit = 0
        prefix = -1
        for i in range(seq.num_blocks - 1):
            seq_block = seq.block(i)
            hash_val = self.compute_hash(seq_block, prefix)
            block_id = self.hash_to_block_id.get(hash_val)
            if block_id is None:
                break
            cache_block = self.blocks[block_id]
            if cache_block.token_ids != seq_block:
                break
            num_blocks_reused += 1
            prefix = hash_val
            self._touch(block_id)
            if block_id in self._free_blocks:
                free_cache_hit += 1

        # Free cache hits still count against free capacity.
        if (
            seq.num_blocks - num_blocks_reused
            > len(self.free_block_ids) - free_cache_hit
        ):
            return -1
        return num_blocks_reused

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """Reserve cached and new blocks before prefill; hashes publish later."""
        prefix = -1
        self.prefix_cache_hit_tokens += num_cached_blocks * self.block_size
        self.prefix_cache_hit_blocks += num_cached_blocks
        seq.num_cached_tokens = self.block_size * num_cached_blocks
        for i in range(num_cached_blocks):
            seq_block = seq.block(i)
            hash_val = self.compute_hash(seq_block, prefix)
            block_id = self.hash_to_block_id[hash_val]
            if block_id in self._free_blocks:
                self._free_blocks.remove(block_id)
                self.used_block_ids.add(block_id)
            prefix = hash_val
            block = self.blocks[block_id]
            block.ref_count += 1
            self._touch(block_id)
            seq.block_table.append(block_id)

        for i in range(num_cached_blocks, seq.num_blocks):
            seq_block = seq.block(i)
            hash_val = self.compute_hash(seq_block, prefix)
            prefix = hash_val
            new_block_id = self._allocate_block()
            self.blocks[new_block_id].update(hash_val, seq_block)
            seq.block_table.append(new_block_id)

    def deallocate(self, seq: Sequence):
        """Release every block referenced by the sequence."""
        seq.num_cached_tokens = 0
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.block_table.clear()

    def trim(self, seq: Sequence):
        """Release blocks beyond the sequence's current logical length."""
        while len(seq.block_table) > seq.num_blocks:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= max(0, seq.num_blocks - len(seq.block_table))

    def may_append(self, seq: Sequence):
        if not self.can_append(seq):
            raise ValueError("insufficient cache blocks")
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """Publish hashes for blocks completed by the scheduled tokens."""
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        prefix = self.blocks[seq.block_table[start - 1]].hash if start != 0 else -1
        for i in range(start, end):
            block = seq.block(i)
            hash_val = self.compute_hash(block, prefix)
            prefix = hash_val
            block_id = seq.block_table[i]
            self._publish_hash(hash_val, block_id)
            self.blocks[block_id].update(hash_val, block)
