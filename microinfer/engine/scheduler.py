from collections import deque

from microinfer.config.engine import EngineConfig
from microinfer.engine.blockmanager import BlockManager
from microinfer.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager):
        self.config = config
        self._inflight = None
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos_token_id
        self.block_size = block_manager.block_size
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if seq.status != SequenceStatus.WAITING or any(
            s.seq_id == seq.seq_id for s in (*self.waiting, *self.running)
        ):
            raise ValueError("request already admitted or finished")
        if not seq.token_ids or seq.max_tokens <= 0 or len(seq) >= self.max_length:
            raise ValueError("prompt/generation length exceeds engine capacity")
        if seq.block_table:
            raise ValueError("new request already owns cache blocks")
        seq.block_size = self.block_size
        self.waiting.append(seq)

    @property
    def max_length(self):
        return min(
            self.config.max_model_len, len(self.block_manager.blocks) * self.block_size
        )

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self._inflight is not None:
            raise RuntimeError("previous schedule has not been postprocessed")
        if self.is_finished():
            return [], True
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                remaining < num_tokens and scheduled_seqs
            ):  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            self._inflight = (scheduled_seqs, True)
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            # Preemption can free enough space to restart a waiting prefill.
            if any(
                self.block_manager.can_allocate(s) >= 0 for s in list(self.waiting)[:1]
            ):
                return self.schedule()
            raise RuntimeError("KV cache cannot make progress")
        self.running.extendleft(reversed(scheduled_seqs))
        self._inflight = (scheduled_seqs, False)
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        if self._inflight != (seqs, is_prefill) or len(seqs) != len(token_ids):
            raise ValueError("postprocess does not match scheduled work")
        if any(type(t) is not int or t < 0 for t in token_ids):
            raise ValueError("invalid sampled token")
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens >= seq.max_tokens
                or len(seq) >= self.max_length
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        self._inflight = None

    def abort(self, seq_id):
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    self.block_manager.deallocate(seq)
                    seq.num_scheduled_tokens = 0
                    seq.status = SequenceStatus.FINISHED
                    if self._inflight is not None:
                        remaining, mode = self._inflight
                        remaining = [s for s in remaining if s.seq_id != seq_id]
                        self._inflight = (remaining, mode) if remaining else None
                    return True
        return False
