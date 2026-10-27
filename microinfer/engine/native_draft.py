"""Speculative draft backed by MicroInfer's paged runner and CUDA Graph."""

from __future__ import annotations

from dataclasses import dataclass

from microinfer.config import SampleParameter
from microinfer.engine.blockmanager import BlockManager
from microinfer.engine.sequence import Sequence
from microinfer.engine.speculative import common_prefix
from microinfer.layer.sampler import Sampler
from microinfer.runner import ModelRunner


@dataclass
class _NativeState:
    seq: Sequence


class NativeDraftSpeculator:
    """A layer-pruned Qwen model running through ModelRunner.

    The draft has its own paged KV cache and uses the same fast paths as the
    target: packed weights, block16 Triton decode, and CUDA Graph capture.
    """

    def __init__(
        self,
        model,
        *,
        block_size: int,
        num_blocks: int,
        max_drafts: int = 8,
        max_num_seqs: int = 8,
        use_cuda_graph: bool = True,
    ):
        if type(max_drafts) is not int or max_drafts <= 0:
            raise ValueError("max_drafts must be a positive integer")
        self.model = model
        self.max_drafts = max_drafts
        self.block_size = block_size
        self.manager = BlockManager(num_blocks, block_size)
        self.runner = ModelRunner(
            model,
            None,
            Sampler(validate=False),
            block_size=block_size,
            device=model.embedding.weight.device,
            use_cuda_graph=use_cuda_graph,
            max_num_seqs=max_num_seqs,
        )
        self.runner.allocate_kv_cache(num_blocks)
        self._states: dict[int, _NativeState] = {}

    def release(self, seq_id: int):
        state = self._states.pop(seq_id, None)
        if state is not None:
            self.manager.deallocate(state.seq)

    def _new_state(self, target) -> _NativeState | None:
        tokens = list(target.token_ids)
        draft = Sequence(
            tokens,
            SampleParameter(
                temperature=1.0,
                max_tokens=1_000_000,
                greedy=True,
                ignore_eos=True,
            ),
        )
        draft.block_size = self.block_size
        if self.manager.can_allocate(draft) == -1:
            return None
        self.manager.allocate(draft, 0)
        state = _NativeState(draft)
        self._states[target.seq_id] = state
        return state

    def _ensure_blocks(self, seq: Sequence) -> bool:
        if len(seq.block_table) > seq.num_blocks:
            self.manager.trim(seq)
        if not self.manager.can_append(seq):
            return False
        self.manager.may_append(seq)
        return True

    def _run_many(self, seqs, is_prefill: bool):
        tokens = self.runner.run(seqs, is_prefill)
        for seq in seqs:
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
        return tokens

    def _prepare(self, target, limit):
        state = self._states.get(target.seq_id)
        if state is None:
            state = self._new_state(target)
            if state is None:
                return None
            scheduled = len(target.token_ids)
            state.seq.num_scheduled_tokens = scheduled
            return state, scheduled, True, []
        seq = state.seq
        target_ids = list(target.token_ids)
        common = common_prefix(seq.token_ids, target_ids)
        start = min(seq.num_cached_tokens, common)
        seq.token_ids = target_ids
        seq.num_tokens = len(target_ids)
        seq.last_token = target_ids[-1]
        if not self._ensure_blocks(seq):
            return None
        scheduled = len(target_ids) - start
        if scheduled <= 0:
            start = max(0, len(target_ids) - 1)
            scheduled = 1
        seq.num_cached_tokens = start
        seq.num_scheduled_tokens = scheduled
        return state, scheduled, scheduled > 1, []

    def propose_batch(self, targets, limits):
        prepared = []
        for target, limit in zip(targets, limits):
            item = self._prepare(target, limit)
            if item is None:
                return [[] for _ in targets]
            state, scheduled, is_prefill, drafts = item
            prepared.append([state, limit, scheduled, is_prefill, drafts])
        prefill = [item for item in prepared if item[2] > 1]
        decode = [item for item in prepared if item[2] == 1]
        if prefill:
            seqs = [item[0].seq for item in prefill]
            tokens = self._run_many(seqs, True)
            for item, token in zip(prefill, tokens):
                seq = item[0].seq
                seq.token_ids.append(token)
                seq.num_tokens += 1
                seq.last_token = token
                item[4].append(token)
        if decode:
            seqs = [item[0].seq for item in decode]
            tokens = self._run_many(seqs, False)
            for item, token in zip(decode, tokens):
                seq = item[0].seq
                seq.token_ids.append(token)
                seq.num_tokens += 1
                seq.last_token = token
                item[4].append(token)

        max_rounds = max(limits, default=0) - 1
        for _ in range(max_rounds):
            active = []
            for item in prepared:
                if len(item[4]) >= item[1]:
                    continue
                if not self._ensure_blocks(item[0].seq):
                    continue
                item[0].seq.num_scheduled_tokens = 1
                active.append(item)
            if not active:
                break
            seqs = [item[0].seq for item in active]
            tokens = self._run_many(seqs, False)
            for item, token in zip(active, tokens):
                seq = item[0].seq
                seq.token_ids.append(token)
                seq.num_tokens += 1
                seq.last_token = token
                item[4].append(token)
        return [item[4] for item in prepared]

    def propose(self, target, limit: int | None = None) -> list[int]:
        limit = self.max_drafts if limit is None else min(limit, self.max_drafts)
        if limit <= 0:
            return []
        return self.propose_batch([target], [limit])[0]
