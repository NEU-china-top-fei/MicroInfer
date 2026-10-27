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
        # Observability; these counters never affect admission decisions.
        self.num_preemptions = 0
        self.num_steps = 0
        self.num_prefill_steps = 0
        self.num_decode_steps = 0
        self.num_mixed_steps = 0
        self.num_spec_steps = 0
        self.num_spec_draft_tokens = 0
        self.num_spec_accepted_tokens = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "num_preemptions": self.num_preemptions,
            "num_steps": self.num_steps,
            "num_prefill_steps": self.num_prefill_steps,
            "num_decode_steps": self.num_decode_steps,
            "num_mixed_steps": self.num_mixed_steps,
            "num_spec_steps": self.num_spec_steps,
            "num_spec_draft_tokens": self.num_spec_draft_tokens,
            "num_spec_accepted_tokens": self.num_spec_accepted_tokens,
            "spec_acceptance_rate": (
                self.num_spec_accepted_tokens / self.num_spec_draft_tokens
                if self.num_spec_draft_tokens
                else None
            ),
        }

    def add(self, seq: Sequence):
        if self.config.debug_mode:
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
        """Schedule one step: running decode first, then waiting prefill.

        Decode requests keep priority, so a long prefill cannot block an active
        generation. Remaining token budget is filled with prefills from the
        waiting queue. LLMEngine executes the two groups separately so decode
        keeps its graph/compiled path.
        """
        if self._inflight is not None:
            raise RuntimeError("previous schedule has not been postprocessed")
        if self.is_finished():
            return [], True

        budget = self.max_num_batched_tokens
        if self.config.mixed_batch:
            scheduled_seqs, budget = self._schedule_decode([], budget)
            scheduled_seqs = self._schedule_prefill(scheduled_seqs, budget)
        else:
            scheduled_seqs = self._schedule_prefill([], budget)
            if not scheduled_seqs:
                scheduled_seqs, _ = self._schedule_decode([], budget)

        if not scheduled_seqs:
            # Preemption can free enough space to restart a waiting prefill.
            if any(
                self.block_manager.can_allocate(s) >= 0 for s in list(self.waiting)[:1]
            ):
                return self.schedule()
            raise RuntimeError("KV cache cannot make progress")

        is_prefill = any(seq.is_prefill for seq in scheduled_seqs)
        self.num_steps += 1
        if is_prefill and len(scheduled_seqs) > sum(
            1 for seq in scheduled_seqs if seq.is_prefill
        ):
            self.num_mixed_steps += 1
        elif is_prefill:
            self.num_prefill_steps += 1
        else:
            self.num_decode_steps += 1
        self._inflight = (scheduled_seqs, is_prefill)
        return scheduled_seqs, is_prefill

    def _schedule_decode(
        self, scheduled_seqs: list[Sequence], budget: int
    ) -> tuple[list[Sequence], int]:
        scheduled: list[Sequence] = []
        for _ in range(len(self.running)):
            if (
                not self.running
                or len(scheduled_seqs) + len(scheduled) >= self.max_num_seqs
                or budget <= 0
            ):
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq is None:
                continue
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            scheduled.append(seq)
            budget -= 1
        # Round-robin: when the token budget cannot cover every running request,
        # the ones served this step move to the back so peers get a turn.
        self.running.extend(scheduled)
        return scheduled_seqs + scheduled, budget

    def _schedule_prefill(
        self, scheduled_seqs: list[Sequence], budget: int
    ) -> list[Sequence]:
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if num_tokens <= 0:
                break
            take = min(num_tokens, budget, self.config.max_num_prefill_tokens)
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = take
            seq.is_prefill = True
            budget -= take
            if seq.num_cached_tokens + take == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            if take < num_tokens:
                # One chunked prefill per step leaves budget for queued
                # prefills to make progress without repeatedly reordering.
                break
        return scheduled_seqs

    def record_spec_attempt(self, seqs: list[Sequence]):
        self.num_spec_steps += 1
        self.num_spec_draft_tokens += sum(len(seq.spec_draft_ids) for seq in seqs)

    def commit_spec(self, seqs: list[Sequence], accepted: list[int]):
        """Commit accepted draft tokens; the caller runs a normal decode next.

        The last accepted draft is left "uncached" so the normal decode step
        processes it and samples the next token with the same kernel as
        non-speculative generation. This keeps partial acceptance exact.
        """
        for seq, count in zip(seqs, accepted):
            drafts = len(seq.spec_draft_ids)
            if count < 0 or count > drafts:
                raise ValueError("invalid speculative acceptance count")
            base = seq.spec_base_len
            seq.token_ids = seq.token_ids[: base + count]
            seq.num_tokens = len(seq.token_ids)
            seq.num_scheduled_tokens = 1 + count
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens = base + count - 1
            seq.num_scheduled_tokens = 0
            seq.spec_draft_ids = ()
            seq.spec_base_len = 0
            seq.is_prefill = False
            seq.last_token = seq.token_ids[-1]
            self.block_manager.trim(seq)
            self.num_spec_accepted_tokens += count
            if (
                seq.num_completion_tokens >= seq.max_tokens
                or len(seq) >= self.max_length
            ):
                seq.finish_reason = "length"
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        if self.config.debug_mode:
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
            if not seq.ignore_eos and token_id == self.eos:
                seq.finish_reason = "eos"
            elif (
                seq.num_completion_tokens >= seq.max_tokens
                or len(seq) >= self.max_length
            ):
                seq.finish_reason = "length"
            if seq.finish_reason is not None:
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
                    seq.finish_reason = "abort"
                    if self._inflight is not None:
                        remaining, mode = self._inflight
                        remaining = [s for s in remaining if s.seq_id != seq_id]
                        self._inflight = (remaining, mode) if remaining else None
                    return True
        return False
