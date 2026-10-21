"""Model execution, attention metadata preparation, and KV cache binding."""

import torch

from microinfer.utils.context import reset_context, set_context
from microinfer.worker.kv_cache import KVCache


class ModelRunner:
    def __init__(self, model, kv_cache, sampler, *, block_size: int, device="cpu"):
        self.model = model
        self.kv_cache = None
        self.sampler = sampler
        self.block_size = block_size
        self.device = device
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

        set_context(
            True,
            indices(cu_q),
            indices(cu_k),
            max((s.num_scheduled_tokens for s in seqs), default=0),
            max(
                (s.num_cached_tokens + s.num_scheduled_tokens for s in seqs), default=0
            ),
            indices(slots),
            None,
            self.prepare_block_tables(seqs)
            if any(s.num_cached_tokens for s in seqs)
            else None,
        )
        return (
            torch.tensor(ids, dtype=torch.int64, device=self.device),
            torch.tensor(positions, dtype=torch.int64, device=self.device),
        )

    def prepare_decode(self, seqs):
        positions = [len(s) - 1 for s in seqs]
        slots = [
            s.block_table[p // self.block_size] * self.block_size + p % self.block_size
            for s, p in zip(seqs, positions)
        ]
        set_context(
            False,
            slot_mapping=torch.tensor(slots, dtype=torch.int32, device=self.device),
            context_lens=torch.tensor(
                [len(s) for s in seqs], dtype=torch.int32, device=self.device
            ),
            block_tables=self.prepare_block_tables(seqs),
        )
        return (
            torch.tensor(
                [s.last_token for s in seqs], dtype=torch.int64, device=self.device
            ),
            torch.tensor(positions, dtype=torch.int64, device=self.device),
        )

    def prepare_sample(self, seqs):
        return torch.tensor(
            [s.temperature for s in seqs], dtype=torch.float32, device=self.device
        )

    def run_model(self, input_ids, positions, is_prefill):
        return self.model.compute_logits(self.model(input_ids, positions))

    @torch.inference_mode()
    def run(self, seqs, is_prefill):
        if not seqs:
            return []
        self.model.eval()
        try:
            ids, positions = (
                self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            )
            logits = self.run_model(ids, positions, is_prefill)
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
