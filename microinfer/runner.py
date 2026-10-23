"""Model execution, attention metadata preparation, and KV cache binding."""

import torch

from microinfer.layer.metadata import AttentionMetadata
from microinfer.utils.context import get_context, reset_context, set_context
from microinfer.worker.kv_cache import KVCache


class ModelRunner:
    def __init__(
        self,
        model,
        kv_cache,
        sampler,
        *,
        block_size: int,
        device="cpu",
        debug_mode: bool = False,
        use_compile: bool = False,
    ):
        if type(debug_mode) is not bool or type(use_compile) is not bool:
            raise ValueError("debug_mode and use_compile must be booleans")
        self.model = model
        self.kv_cache = None
        self.sampler = sampler
        self.block_size = block_size
        self.device = device
        self.debug_mode = debug_mode
        self.use_compile = use_compile
        self._forward = None if self.model is None else self.model.forward
        self._compiled_forward = None
        if self.use_compile and self.model is not None:
            # Build setup-only state (the 40k-row RoPE table) before Dynamo sees
            # it; tracing table construction is expensive and shape-specialized.
            rotary = getattr(self.model, "rotary", None)
            if rotary is not None:
                rotary.ensure(torch.device(self.device))
            # Compile decode only: prefill has variable-length Python loops and
            # gains little, while decode dominates token latency.
            self._compiled_forward = torch.compile(
                self.model.forward, dynamic=True, mode="default"
            )
        if kv_cache is not None:
            self.bind_kv_cache(kv_cache)

    def bind_kv_cache(self, cache: KVCache):
        """Bind all layer views once, or explicitly rebind for a new cache pool.

        Call after moving the model to its final dtype/device; cache views are
        runtime storage, not model parameters or persistent state_dict buffers.
        """
        layers = self.model.transformer_layers
        if len(layers) != cache.num_layers or cache.block_size != self.block_size:
            raise ValueError("cache layer count/block size does not match runner")
        views = [cache.layer(i) for i in range(len(layers))]
        for layer, view in zip(layers, views):
            attention = layer.attention
            if view.keys.shape[2:] != (attention.num_kv_head, attention.head_dim):
                raise ValueError("cache heads do not match model")
            weight = attention.k_proj.weight
            if view.keys.device != weight.device or view.keys.dtype != weight.dtype:
                raise ValueError("cache dtype/device does not match model")
        for layer, view in zip(layers, views):
            layer.attention.bind_cache(view)
        self.kv_cache = cache

    def allocate_kv_cache(self, num_blocks: int):
        """Allocate the runtime pool using the model's final dtype/device."""
        config = self.model.config
        weight = self.model.embedding.weight
        cache = KVCache(
            config.num_hidden_layer,
            num_blocks,
            self.block_size,
            config.num_kv_head,
            config.head_dim,
            dtype=weight.dtype,
            device=weight.device,
        )
        self.bind_kv_cache(cache)
        return cache

    def prepare_block_tables(self, seqs):
        width = max((len(seq.block_table) for seq in seqs), default=0)
        return torch.tensor(
            [seq.block_table + [-1] * (width - len(seq.block_table)) for seq in seqs],
            dtype=torch.int32,
            device=self.device,
        ).reshape(len(seqs), width)

    def prepare_prefill(self, seqs):
        ids, positions, slots, cu_q, cu_k = [], [], [], [0], [0]
        for seq in seqs:
            start, end = (
                seq.num_cached_tokens,
                seq.num_cached_tokens + seq.num_scheduled_tokens,
            )
            if not 0 <= start < end <= len(seq):
                raise ValueError("invalid scheduled prefill range")
            ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_q.append(cu_q[-1] + end - start)
            cu_k.append(cu_k[-1] + end)
            slots.extend(
                seq.block_table[p // self.block_size] * self.block_size
                + p % self.block_size
                for p in range(start, end)
            )

        def indices(values):
            return torch.tensor(values, dtype=torch.int32, device=self.device)

        cu_q_tensor = indices(cu_q)
        cu_k_tensor = indices(cu_k)
        slot_tensor = indices(slots)
        max_seqlen_q = max((s.num_scheduled_tokens for s in seqs), default=0)
        max_seqlen_k = max(
            (s.num_cached_tokens + s.num_scheduled_tokens for s in seqs), default=0
        )
        block_tables = (
            self.prepare_block_tables(seqs)
            if any(s.num_cached_tokens for s in seqs)
            else None
        )
        metadata = AttentionMetadata.build(
            is_prefill=True,
            cu_seqlens_q=cu_q_tensor,
            cu_seqlens_k=cu_k_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_tensor,
            block_tables=block_tables,
            validate_cache_write=self.debug_mode,
        )
        set_context(
            True,
            cu_q_tensor,
            cu_k_tensor,
            max_seqlen_q,
            max_seqlen_k,
            slot_tensor,
            None,
            block_tables,
            metadata,
        )
        input_ids = torch.tensor(ids, dtype=torch.int64, device=self.device)
        position_ids = torch.tensor(positions, dtype=torch.int64, device=self.device)
        if self.debug_mode:
            self.validate_step(seqs, input_ids, position_ids, True)
        return input_ids, position_ids

    def prepare_decode(self, seqs):
        positions = [len(s) - 1 for s in seqs]
        slots = [
            s.block_table[p // self.block_size] * self.block_size + p % self.block_size
            for s, p in zip(seqs, positions)
        ]
        slot_tensor = torch.tensor(slots, dtype=torch.int32, device=self.device)
        context_tensor = torch.tensor(
            [len(s) for s in seqs], dtype=torch.int32, device=self.device
        )
        block_tables = self.prepare_block_tables(seqs)
        metadata = AttentionMetadata.build(
            is_prefill=False,
            slot_mapping=slot_tensor,
            context_lens=context_tensor,
            block_tables=block_tables,
            validate_cache_write=self.debug_mode,
        )
        set_context(
            False,
            slot_mapping=slot_tensor,
            context_lens=context_tensor,
            block_tables=block_tables,
            metadata=metadata,
        )
        input_ids = torch.tensor(
            [s.last_token for s in seqs], dtype=torch.int64, device=self.device
        )
        position_ids = torch.tensor(positions, dtype=torch.int64, device=self.device)
        if self.debug_mode:
            self.validate_step(seqs, input_ids, position_ids, False)
        return input_ids, position_ids

    def validate_step(self, seqs, input_ids, positions, is_prefill):
        """Validate scheduler/runner invariants; only called in debug_mode."""
        context = get_context()
        if context.is_prefill is not is_prefill:
            raise ValueError("context mode does not match scheduled work")
        if input_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError("input_ids and positions must be 1-D")
        if input_ids.numel() != positions.numel() or input_ids.numel() == 0:
            raise ValueError("input_ids/positions must be nonempty and equal length")
        if not seqs:
            raise ValueError("a scheduled batch must contain requests")

        block_size = self.block_size
        num_blocks = getattr(self.kv_cache, "num_blocks", None)
        slots = context.slot_mapping
        if slots is None or slots.ndim != 1 or slots.numel() != input_ids.numel():
            raise ValueError("slot_mapping must match scheduled tokens")
        slot_list = [int(s) for s in slots.tolist()]
        if any(s < 0 for s in slot_list):
            raise ValueError("slot_mapping must be nonnegative")
        if len(set(slot_list)) != len(slot_list):
            raise ValueError("slot_mapping contains duplicate slots")
        if num_blocks is not None and any(
            s >= num_blocks * block_size for s in slot_list
        ):
            raise ValueError("slot_mapping exceeds KV capacity")

        position_list = [int(p) for p in positions.tolist()]
        if any(p < 0 for p in position_list):
            raise ValueError("positions must be nonnegative")
        max_pos = getattr(getattr(self.model, "config", None), "max_pos_embed", None)
        if max_pos is not None and any(p >= max_pos for p in position_list):
            raise ValueError("position exceeds model max_pos_embed")
        token_list = [int(t) for t in input_ids.tolist()]
        if any(t < 0 for t in token_list):
            raise ValueError("input_ids must be nonnegative")
        vocab = getattr(getattr(self.model, "config", None), "vocab_size", None)
        if vocab is not None and any(t >= vocab for t in token_list):
            raise ValueError("input_ids exceed model vocabulary")

        def check_blocks(seq, required):
            if len(seq.block_table) < required:
                raise ValueError("block table does not cover scheduled tokens")
            for block_id in seq.block_table[:required]:
                block_id = int(block_id)
                if block_id < 0:
                    raise ValueError("block table contains a negative block id")
                if num_blocks is not None and block_id >= num_blocks:
                    raise ValueError("block id exceeds KV pool")

        if is_prefill:
            cu_q, cu_k = context.cu_seqlens_q, context.cu_seqlens_k
            if cu_q is None or cu_k is None:
                raise ValueError("prefill requires cu_seqlens_q/cu_seqlens_k")
            q = [int(v) for v in cu_q.tolist()]
            k = [int(v) for v in cu_k.tolist()]
            if len(q) != len(seqs) + 1 or len(k) != len(seqs) + 1:
                raise ValueError("cu_seqlens must contain num_requests + 1 entries")
            if q[0] != 0 or k[0] != 0 or q[-1] != input_ids.numel():
                raise ValueError("invalid cu_seqlens endpoints")
            q_lens, k_lens = [], []
            for i, seq in enumerate(seqs):
                start = seq.num_cached_tokens
                end = start + seq.num_scheduled_tokens
                q_len, k_len = q[i + 1] - q[i], k[i + 1] - k[i]
                if q_len <= 0:
                    raise ValueError("prefill query length must be positive")
                if q_len != seq.num_scheduled_tokens:
                    raise ValueError("query length does not match scheduled tokens")
                if k_len != end:
                    raise ValueError("KV length does not match cached+scheduled tokens")
                if len(seq) < end:
                    raise ValueError("scheduled tokens exceed sequence length")
                required = (end + block_size - 1) // block_size
                check_blocks(seq, required)
                if position_list[q[i] : q[i + 1]] != list(range(start, end)):
                    raise ValueError("positions do not match prefill range")
                expected_slots = [
                    seq.block_table[p // block_size] * block_size + p % block_size
                    for p in range(start, end)
                ]
                if slot_list[q[i] : q[i + 1]] != expected_slots:
                    raise ValueError("slot_mapping does not match block table")
                q_lens.append(q_len)
                k_lens.append(k_len)
            if q[-1] != sum(q_lens) or k[-1] != sum(k_lens):
                raise ValueError("cu_seqlens totals do not match")
            if context.max_seqlen_q != max(q_lens):
                raise ValueError("max_seqlen_q does not match cu_seqlens_q")
            if context.max_seqlen_k != max(k_lens):
                raise ValueError("max_seqlen_k does not match cu_seqlens_k")
            if any(seq.num_cached_tokens for seq in seqs) and context.block_tables is None:
                raise ValueError("cached prefixes require block_tables")
        else:
            if len(seqs) != input_ids.numel():
                raise ValueError("decode needs exactly one token per request")
            if context.context_lens is None or context.context_lens.numel() != len(seqs):
                raise ValueError("decode requires one context length per request")
            if context.block_tables is None:
                raise ValueError("decode requires block_tables")
            lens = [int(v) for v in context.context_lens.tolist()]
            for i, (seq, context_len) in enumerate(zip(seqs, lens)):
                if context_len != len(seq) or context_len <= 0:
                    raise ValueError("context length does not match sequence length")
                if position_list[i] != context_len - 1:
                    raise ValueError("decode position does not match context length")
                required = (context_len + block_size - 1) // block_size
                check_blocks(seq, required)
                position = context_len - 1
                expected_slot = (
                    seq.block_table[position // block_size] * block_size
                    + position % block_size
                )
                if slot_list[i] != expected_slot:
                    raise ValueError("decode slot does not match block table")

        if context.block_tables is not None:
            if (
                context.block_tables.ndim != 2
                or context.block_tables.shape[0] != len(seqs)
            ):
                raise ValueError("block_tables must be [num_requests, max_blocks]")
            table = context.block_tables.tolist()
            for i, seq in enumerate(seqs):
                context_len = (
                    seq.num_cached_tokens + seq.num_scheduled_tokens
                    if is_prefill
                    else len(seq)
                )
                required = (context_len + block_size - 1) // block_size
                if required > len(table[i]):
                    raise ValueError("block table does not cover context")
                for block_id in table[i][:required]:
                    block_id = int(block_id)
                    if block_id < 0:
                        raise ValueError("block table contains a negative block id")
                    if num_blocks is not None and block_id >= num_blocks:
                        raise ValueError("block id exceeds KV pool")

    def prepare_sample(self, seqs):
        return torch.tensor(
            [s.temperature for s in seqs], dtype=torch.float32, device=self.device
        )

    def _forward_for(self, is_prefill):
        if self._compiled_forward is not None and not is_prefill:
            return self._compiled_forward
        return self._forward

    def run_model(self, input_ids, positions, is_prefill, metadata=None):
        if metadata is None:
            metadata = AttentionMetadata.from_context()
        hidden = self._forward_for(is_prefill)(input_ids, positions, metadata)
        return self.model.compute_logits(hidden, metadata)

    @torch.inference_mode()
    def run(self, seqs, is_prefill):
        if not seqs:
            return []
        self.model.eval()
        try:
            ids, positions = (
                self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            )
            metadata = get_context().metadata
            logits = self.run_model(ids, positions, is_prefill, metadata)
            greedy = torch.tensor(
                [getattr(s, "greedy", False) for s in seqs], device=logits.device
            )
            if greedy.all():
                tokens = logits.argmax(-1)
            else:
                tokens = self.sampler(logits, self.prepare_sample(seqs))
                tokens = torch.where(greedy, logits.argmax(-1), tokens)
            return tokens.tolist()
        finally:
            reset_context()
