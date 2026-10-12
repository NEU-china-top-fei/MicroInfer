"""M01: nano-vllm scheduling contract (prefill first, homogeneous batches).

Reference: nano-vllm bb823b3. Constructor injection is a MicroInfer test seam.
Do not call schedule on an empty engine; use is_finished first.
"""

from microinfer.config.engine import EngineConfig
from microinfer.engine.blockmanager import BlockManager
from microinfer.engine.sequence import Sequence, SequenceStatus
from collections import deque


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager
        # TODO: waiting/running deques, max_num_seqs, token budget, eos, block_size.
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()  # contain seq_id
        self.max_num_seqs = config.max_num_seqs
        self.token_budget = config.max_num_batched_tokens
        self.eos = config.eos_token_id
        self.block_sise = block_manager.block_size

    def add(self, seq: Sequence) -> None:
        """Append to waiting. Admission guards are optional MicroInfer extensions."""
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        """Both waiting and running are empty."""
        return len(self.waiting) == 0 and len(self.running) == 0

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Try waiting prefills first. Return (seqs, is_prefill).

        Only the first selected prefill may be chunked; keep an incomplete one
        in waiting. Move a completing prefill into running before execution.
        If any prefill is selected, return without scheduling decode.
        Otherwise schedule one query per running request, up to max_num_seqs.
        Decode does NOT use max_num_batched_tokens as an additional cap.
        On decode pressure, pop and preempt the tail of remaining running work.
        Do not publish hashes or advance cached tokens here.
        """
        seq_scheduled = []
        token_scheduled = 0
        num_seq_scheduled = 0
        is_prefill = True
        # breakpoint()
        # prefill
        while self.waiting and num_seq_scheduled < self.max_num_seqs:
            if self.token_budget == token_scheduled:
                break
            new_seq = self.waiting[0]
            # breakpoint()
            if not new_seq.block_table:
                # empty block table means first time to prefill
                num_cache_block = self.block_manager.can_allocate(new_seq)
                new_seq.num_cached_tokens = num_cache_block * self.block_sise
                new_token = new_seq.num_tokens - self.block_sise * num_cache_block
            else:
                new_token = new_seq.num_tokens - new_seq.num_cached_tokens
                num_cache_block = new_seq.num_cached_tokens // self.block_sise
            if new_token + token_scheduled > self.token_budget and seq_scheduled:
                # not the first seq need split
                break
            true_token = min(new_token, self.token_budget - token_scheduled)
            num_seq_scheduled += 1
            token_scheduled += true_token

            new_seq.num_scheduled_tokens = true_token
            if not new_seq.block_table:
                self.block_manager.allocate(new_seq, num_cache_block)
            if true_token + new_seq.num_cached_tokens == new_seq.num_tokens:
                # prefill finish
                self.waiting.popleft()
                self.running.append(new_seq)
                new_seq.status = SequenceStatus.RUNNING
            seq_scheduled.append(new_seq)
        if seq_scheduled:
            return seq_scheduled, is_prefill

        # decode
        is_prefill = False
        while self.running and num_seq_scheduled < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.popleft())
                else:
                    self.preempt(seq)
            else:
                # can handle append
                token_scheduled += 1
                num_seq_scheduled += 1
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                seq_scheduled.append(seq)
        self.running.extendleft(reversed(seq_scheduled))
        return seq_scheduled, is_prefill

    def preempt(self, seq: Sequence) -> None:
        """Caller has already removed seq from running. Release blocks, mark
        WAITING/is_prefill, preserve tokens and appendleft to waiting.
        preempt:抢占
        """
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.waiting.appendleft(seq)

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool
    ) -> None:
        """Positional correspondence: one sampled token per scheduled sequence.

        hash_blocks -> advance cached count -> clear scheduled count.
        Discard the sampled token for an incomplete prefill chunk. Otherwise
        append it; EOS/max_tokens finishes and releases/removes the sequence.
        Return None; LLMEngine gathers finished sequences after this call.
        """
        for token_id, seq in zip(token_ids, seqs):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_prompt_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or (
                seq.num_completion_tokens == seq.max_tokens
            ):  # finish or truncated
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def abort(self, seq_id: int) -> bool:
        """Optional MicroInfer extension, not part of the pinned nano-vllm API."""
        raise NotImplementedError("extension: Scheduler.abort")
