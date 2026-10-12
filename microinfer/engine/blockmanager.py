from microinfer.config import SampleParameter
import xxhash
from collections import deque
import numpy as np
from microinfer.engine.sequence import Sequence

"""
blockmanager<---------->sequence
  |
  |__block_size
  |__num_blocks
  |__free_blocks
  |__used_blocks
  |__map_from_hash_to_block
  |__blocks->list of (block)
                        |
                        |
                        |__block_id
                        |__hash
                        |__tokenids
                        |__refcnt
"""


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        reset to the state of recently allocated
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        prefix: the hash val of the whole prefix block
        """
        hash_fuc = xxhash.xxh64()
        if prefix != -1:
            # this has prefix
            # update just update the internal state
            hash_fuc.update(prefix.to_bytes(8, "little"))

        hash_fuc.update(np.array(token_ids).tobytes())

        return hash_fuc.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        # you can not change the size of dict while iteratition
        old_hashes = [h for h, bid in self.hash_to_block_id.items() if bid == block_id]

        for h in old_hashes:
            del self.hash_to_block_id[h]

        self.used_block_ids.add(block_id)
        block = self.blocks[block_id]
        block.reset()

        return block_id

    def _deallocate_block(self, block_id: int):
        """
        keep kv and map when ref_cnt=0,only when it needs to be reused clean it
        """
        # block = self.blocks[block_id]
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        # del self.hash_to_block_id[block.hash]

    def can_allocate(self, seq: Sequence) -> int:
        """
        compute how many blocks can be reused in this sequence(return -1 if overflow)
        we need to reserve the compute of last block to avoid the cached token excceed the true length
        and deal with hash collision
        """
        num_seq_blocks = seq.num_blocks
        num_blocks_reused = 0
        free_cache_hit = 0
        prefix = -1
        for i in range(num_seq_blocks - 1):
            seq_block = seq.block(i)
            hash_val = self.compute_hash(seq_block, prefix)
            block_id = self.hash_to_block_id.get(hash_val)
            if block_id is None:
                break
            else:
                cache_block = self.blocks[block_id]
                if cache_block.token_ids != seq_block:
                    break
                num_blocks_reused += 1
                prefix = hash_val
                if block_id in self.free_block_ids:
                    free_cache_hit += 1

        # compute whether the rest free block can hold
        # Free cache hits must still count against free capacity
        if (
            seq.num_blocks - num_blocks_reused
            > len(self.free_block_ids) - free_cache_hit
        ):
            return -1
        return num_blocks_reused

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        USED FOR PREFILL

        carry out the true allocate on both reusable and new block
        (
            but just allocate,we don't carry out the true compute so we can't update the table immediately
            so we can only update the manager index after compute using special function
        )

        """
        prefix = -1
        seq.num_cached_tokens = (
            self.block_size * num_cached_blocks
        )  # we need to cover previou value
        for i in range(num_cached_blocks):
            seq_block = seq.block(i)
            hash_val = self.compute_hash(seq_block, prefix)
            block_id = self.hash_to_block_id.get(hash_val)
            if block_id in self.free_block_ids:
                # cache hit the zero ref_cnt block
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            prefix = hash_val
            block = self.blocks[block_id]
            block.ref_count += 1
            seq.block_table.append(block_id)
            # seq.num_cached_tokens += self.block_size
        for i in range(num_cached_blocks, seq.num_blocks):
            seq_block = seq.block(i)
            hash_val = self.compute_hash(seq_block, prefix)
            prefix = hash_val
            new_block_id = self._allocate_block()
            new_block = self.blocks[new_block_id]
            new_block.update(hash_val, seq_block)
            # self.hash_to_block_id[hash_val] = new_block_id
            seq.block_table.append(new_block_id)

    def deallocate(self, seq: Sequence):
        """
        traverse the true block table
        """
        seq.num_cached_tokens = 0
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                # allocate and deallocate both should maintain reference count
                self._deallocate_block(block_id)
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # we just decode a new token to find that we need another block

        # means that we need to allocate new block and have free block
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
        """
        equal to 
            need_blocks = 1 if len(seq)%block_size==1 else 0
            return len(free_block_ids) >= need_blocks
        """

    def may_append(self, seq: Sequence):
        """
        USED FOR DECODE
        """
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        hash the block of this sequence after cached blocks (after some num of token be scheduled)
        """
        # start_block_id = seq.block_table[-1] if len(seq.block_table) != 0 else -1
        # can be the situation that a scheduled token can't fit in a block
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        prefix = self.blocks[seq.block_table[start - 1]].hash if start != 0 else -1
        for i in range(start, end):
            block = seq.block(i)
            hash_val = self.compute_hash(block, prefix)
            prefix = hash_val
            block_id = seq.block_table[i]
            self.hash_to_block_id[hash_val] = block_id
            self.blocks[block_id].update(
                hash_val, block
            )  # update because the block maybe unfilled
