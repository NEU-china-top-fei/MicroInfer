"""Draft proposers for greedy speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def common_prefix(left, right) -> int:
    """Length of the shared prefix of two token sequences."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


class NgramSpeculator:
    """Propose a continuation from the last repeated n-gram.

    The lookup scans backward and skips the current occurrence. It needs no
    model: prompt repetition and self-repetition are cheap signals, and a
    wrong draft is rejected by one target forward.
    """

    def __init__(self, ngram: int = 3, max_drafts: int = 8):
        if type(ngram) is not int or ngram <= 0:
            raise ValueError("ngram must be a positive integer")
        if type(max_drafts) is not int or max_drafts <= 0:
            raise ValueError("max_drafts must be a positive integer")
        self.ngram = ngram
        self.max_drafts = max_drafts

    def propose(self, seq_or_tokens, limit: int | None = None) -> list[int]:
        token_ids = (
            seq_or_tokens.token_ids
            if hasattr(seq_or_tokens, "token_ids")
            else seq_or_tokens
        )
        limit = self.max_drafts if limit is None else min(limit, self.max_drafts)
        if limit <= 0:
            return []
        length = len(token_ids)
        if length <= self.ngram:
            return []
        key = tuple(token_ids[-self.ngram :])
        # Start before the current occurrence of the key.
        for index in range(length - self.ngram - 1, -1, -1):
            if tuple(token_ids[index : index + self.ngram]) == key:
                start = index + self.ngram
                return list(token_ids[start : start + limit])
        return []


@dataclass
class _DraftState:
    ids: tuple[int, ...]
    cache: object
    next_logits: object


class DraftModelSpeculator:
    """HF causal-LM draft with an incremental per-sequence KV cache."""

    def __init__(self, model, *, max_drafts: int = 8, vocab_size=None):
        if type(max_drafts) is not int or max_drafts <= 0:
            raise ValueError("max_drafts must be a positive integer")
        if vocab_size is not None and model.config.vocab_size != vocab_size:
            raise ValueError("draft and target model vocab sizes differ")
        self.model = model.eval()
        self.max_drafts = max_drafts
        self.device = next(model.parameters()).device
        self._states: dict[int, _DraftState] = {}

    def release(self, seq_id: int):
        self._states.pop(seq_id, None)

    def _forward(self, token_ids, cache):
        input_ids = torch.tensor(
            [list(token_ids)], dtype=torch.long, device=self.device
        )
        output = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )
        return output.logits[0, -1], output.past_key_values

    def _reconcile(self, seq_id, target_ids):
        from transformers.cache_utils import DynamicCache

        state = self._states.get(seq_id)
        if state is None:
            cache = DynamicCache()
            logits, cache = self._forward(target_ids, cache)
            state = _DraftState(tuple(target_ids), cache, logits)
            self._states[seq_id] = state
            return state

        common = common_prefix(state.ids, target_ids)
        if common == len(target_ids) and len(state.ids) == len(target_ids):
            return state
        if common < len(state.ids):
            state.cache.crop(-(len(state.ids) - common))
            state.ids = state.ids[:common]
            state.next_logits = None
        if common == len(target_ids):
            # The target is a prefix of the cached draft sequence. Rewind one
            # token so the cache contains exactly target[:-1].
            state.cache.crop(-1)
            state.ids = state.ids[:-1]
            suffix = target_ids[-1:]
        else:
            suffix = target_ids[common:]
        if suffix:
            logits, state.cache = self._forward(suffix, state.cache)
            state.ids = tuple(target_ids)
            state.next_logits = logits
        return state

    def propose(self, seq_or_tokens, limit: int | None = None) -> list[int]:
        if not hasattr(seq_or_tokens, "seq_id"):
            raise TypeError("draft model requires a Sequence")
        seq = seq_or_tokens
        limit = self.max_drafts if limit is None else min(limit, self.max_drafts)
        if limit <= 0:
            return []
        with torch.inference_mode():
            state = self._reconcile(seq.seq_id, tuple(seq.token_ids))
            drafts = []
            logits = state.next_logits
            for _ in range(limit):
                token = int(logits.argmax())
                drafts.append(token)
                state.ids = state.ids + (token,)
                logits, state.cache = self._forward([token], state.cache)
                state.next_logits = logits
        return drafts
