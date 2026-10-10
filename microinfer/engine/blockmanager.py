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
            hash_fuc.update(prefix.to_bytes(8, "little"))

        hash_fuc.update(np.array(token_ids).tobytes())

        return hash_fuc.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        self.used_block_ids.add(block_id)
        block = self.blocks[block_id]
        block.reset()
        return block_id

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        del self.hash_to_block_id[block.hash]

    def can_allocate(self, seq: Sequence) -> int:
        """
        compute how many blocks can be reused in this sequence
        """
        num_seq_blocks = seq.num_blocks
        num_blocks_reused = 0
        prefix = -1
        for i in range(num_seq_blocks):
            seq_block = seq.block(i)
            hash_val = self.compute_hash(seq_block, prefix)
            if self.hash_to_block_id.get(hash_val) is None:
                break
            else:
                num_blocks_reused += 1
                prefix = hash_val
        return num_blocks_reused

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        carry out the true allocate on both reusable and new block
        """
        prefix = -1
        for i in range(num_cached_blocks):
            seq_block = seq.block(i)
            hash_val = self.compute_hash(seq_block, prefix)
            block_id = self.hash_to_block_id.get(hash_val)
            prefix = hash_val
            block = self.blocks[block_id]
            block.ref_count += 1
            seq.block_table.append(block_id)
            seq.num_cached_tokens += self.block_size
        for i in range(num_cached_blocks, seq.num_blocks):
            seq_block = seq.block(i)
            hash_val = self.compute_hash(seq_block, prefix)
            prefix = hash_val
            new_block_id = self._allocate_block()
            new_block = self.blocks[new_block_id]
            new_block.update(hash_val, seq_block)
            self.hash_to_block_id[hash_val] = new_block_id
            seq.block_table.append(new_block_id)

    def deallocate(self, seq: Sequence):
        seq.num_cached_tokens = 0
        for i in range(seq.num_blocks):
            block_id = seq.block_table[i]
            self._deallocate_block(block_id)
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # def hash_blocks(self, seq: Sequence):
    #     """
    #     hash the block of this sequence after cached blocks
    #     """
    #     start_block_id = seq.block_table[-1] if len(seq.block_table) != 0 else -1
