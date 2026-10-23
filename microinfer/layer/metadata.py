"""Attention metadata built once per engine step.

Tensors stay on device for kernels. CPU lists are lazy: the flash paths never
materialize them, while the reference path and no-prefix prefill build them once
per step instead of once per layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import torch


@dataclass
class AttentionMetadata:
    is_prefill: bool
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    direct_prefill: bool = False
    validate_cache_write: bool = True
    _cu_q_cpu: tuple[int, ...] | None = field(default=None, repr=False)
    _context_lens_cpu: tuple[int, ...] | None = field(default=None, repr=False)
    _q_lens_cpu: tuple[int, ...] | None = field(default=None, repr=False)
    _slot_mapping_long: torch.Tensor | None = field(default=None, repr=False)
    _slot_block_size: int | None = field(default=None, repr=False)
    _slot_blocks: torch.Tensor | None = field(default=None, repr=False)
    _slot_offsets: torch.Tensor | None = field(default=None, repr=False)

    @classmethod
    def build(
        cls,
        *,
        is_prefill: bool,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        slot_mapping=None,
        context_lens=None,
        block_tables=None,
        validate_cache_write: bool = True,
    ) -> AttentionMetadata:
        return cls(
            is_prefill=is_prefill,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            direct_prefill=is_prefill and block_tables is None,
            validate_cache_write=validate_cache_write,
        )

    @property
    def is_decode(self) -> bool:
        return not self.is_prefill

    @property
    def slot_mapping_long(self) -> torch.Tensor:
        if self.slot_mapping is None:
            raise ValueError("metadata has no slot_mapping")
        if self._slot_mapping_long is None:
            self._slot_mapping_long = self.slot_mapping.long()
        return self._slot_mapping_long

    def slot_indices(self, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Physical block/offset tensors, computed once per step and layer cache."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if self._slot_block_size != block_size:
            slots = self.slot_mapping_long
            self._slot_blocks = torch.div(slots, block_size, rounding_mode="floor")
            self._slot_offsets = slots - self._slot_blocks * block_size
            self._slot_block_size = block_size
        return self._slot_blocks, self._slot_offsets

    @property
    def cu_q_cpu(self) -> tuple[int, ...]:
        if self._cu_q_cpu is None:
            if self.cu_seqlens_q is not None:
                self._cu_q_cpu = tuple(int(x) for x in self.cu_seqlens_q.tolist())
            elif self.context_lens is not None:
                self._cu_q_cpu = tuple(range(self.context_lens.numel() + 1))
            else:
                self._cu_q_cpu = (0,)
        return self._cu_q_cpu

    @property
    def context_lens_cpu(self) -> tuple[int, ...]:
        if self._context_lens_cpu is None:
            if self.context_lens is not None:
                self._context_lens_cpu = tuple(
                    int(x) for x in self.context_lens.tolist()
                )
            elif self.cu_seqlens_k is not None:
                lengths = [int(x) for x in self.cu_seqlens_k.tolist()]
                self._context_lens_cpu = tuple(b - a for a, b in pairwise(lengths))
            else:
                self._context_lens_cpu = ()
        return self._context_lens_cpu

    @property
    def q_lens_cpu(self) -> tuple[int, ...]:
        if self._q_lens_cpu is None:
            cu = self.cu_q_cpu
            self._q_lens_cpu = tuple(b - a for a, b in pairwise(cu))
        return self._q_lens_cpu
