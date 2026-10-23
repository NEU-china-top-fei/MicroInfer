"""Per-execution metadata, isolated with ContextVar and scoped restoration."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    metadata: object | None = None


_EMPTY = Context()
_CONTEXT = ContextVar("microinfer_context", default=_EMPTY)


def get_context():
    return _CONTEXT.get()


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    metadata=None,
):
    _CONTEXT.set(
        Context(
            is_prefill,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            block_tables,
            metadata,
        )
    )


def reset_context():
    _CONTEXT.set(Context())


@contextmanager
def use_context(context):
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)
