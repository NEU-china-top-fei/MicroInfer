"""Model execution, attention metadata preparation, and KV cache binding."""

import torch

from microinfer.layer.metadata import AttentionMetadata
from microinfer.worker.kv_cache import KVCache

_DECODE_SCALARS = (
    ("input_ids", torch.int64),
    ("positions", torch.int64),
    ("slot_mapping", torch.int32),
    ("context_lens", torch.int32),
)


class _DecodeBuffers:
    """Pinned staging + device buffers for the decode hot path."""

    def __init__(self, device, pin_memory: bool):
        self.device = device
        self.pin_memory = pin_memory
        self.capacity = 0
        self.width = 0
        self.cpu = {}
        self.device_tensors = {}

    def ensure(self, batch: int, width: int):
        if batch <= self.capacity and width <= self.width:
            return
        capacity = max(batch, self.capacity)
        width = max(width, self.width)
        for name, dtype in _DECODE_SCALARS:
            self.cpu[name] = torch.empty(
                capacity, dtype=dtype, pin_memory=self.pin_memory
            )
            self.device_tensors[name] = torch.empty(
                capacity, dtype=dtype, device=self.device
            )
        self.cpu["block_tables"] = torch.empty(
            (capacity, width),
            dtype=torch.int32,
            pin_memory=self.pin_memory,
        )
        self.device_tensors["block_tables"] = torch.empty(
            (capacity, width), dtype=torch.int32, device=self.device
        )
        self.capacity = capacity
        self.width = width


class _GraphBucket:
    def __init__(self, batch, buffers, metadata, graph, logits):
        self.batch = batch
        self.buffers = buffers
        self.metadata = metadata
        self.graph = graph
        self.logits = logits


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
        use_cuda_graph: bool = False,
        max_num_seqs: int = 16,
    ):
        if type(debug_mode) is not bool or type(use_compile) is not bool:
            raise ValueError("debug_mode and use_compile must be booleans")
        if type(use_cuda_graph) is not bool:
            raise ValueError("use_cuda_graph must be a boolean")
        if use_compile and use_cuda_graph:
            raise ValueError("use_compile and use_cuda_graph are mutually exclusive")
        self.model = model
        self.kv_cache = None
        self.sampler = sampler
        self.block_size = block_size
        self.device = device
        self.debug_mode = debug_mode
        self.use_compile = use_compile
        self.use_cuda_graph = use_cuda_graph
        self.max_num_seqs = max_num_seqs
        self._forward = None if self.model is None else self.model.forward
        self._compiled_forward = None
        self._decode_buffers = None
        self._pad_block_id = None
        self._graph_buckets: list[_GraphBucket] = []
        self._graph_pool = None
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
        # A graph holds pointers into the old pool.
        self._graph_buckets = []
        self._graph_pool = None

    def allocate_kv_cache(self, num_blocks: int):
        """Allocate the runtime pool using the model's final dtype/device."""
        config = self.model.config
        weight = self.model.embedding.weight
        # One extra block is never handed to the scheduler; padded graph rows
        # write their KV there instead of corrupting a live request.
        extra = 1 if self.use_cuda_graph else 0
        self._pad_block_id = num_blocks if self.use_cuda_graph else None
        cache = KVCache(
            config.num_hidden_layer,
            num_blocks + extra,
            self.block_size,
            config.num_kv_head,
            config.head_dim,
            dtype=weight.dtype,
            device=weight.device,
        )
        self.bind_kv_cache(cache)
        if self.use_cuda_graph:
            self.capture_cuda_graphs()
        return cache

    def _decode_forward(self, input_ids, positions, metadata):
        hidden = self._forward(input_ids, positions, metadata)
        return self.model.compute_logits(hidden, metadata)

    @staticmethod
    def _copy_decode_buffers(buffers: "_DecodeBuffers", batch: int):
        non_blocking = buffers.pin_memory
        for name, _ in _DECODE_SCALARS:
            buffers.device_tensors[name][:batch].copy_(
                buffers.cpu[name][:batch], non_blocking=non_blocking
            )
        buffers.device_tensors["block_tables"][:batch].copy_(
            buffers.cpu["block_tables"][:batch], non_blocking=non_blocking
        )

    def _make_decode_metadata(self, buffers: "_DecodeBuffers", batch: int):
        return AttentionMetadata.build(
            is_prefill=False,
            slot_mapping=buffers.device_tensors["slot_mapping"][:batch],
            context_lens=buffers.device_tensors["context_lens"][:batch],
            block_tables=buffers.device_tensors["block_tables"][:batch],
            validate_cache_write=False,
        )

    def _fill_pad_buffers(self, buffers: "_DecodeBuffers", batch: int):
        assert self._pad_block_id is not None
        cpu = buffers.cpu
        block_tables = cpu["block_tables"]
        block_tables.fill_(-1)
        for i in range(batch):
            cpu["input_ids"][i] = 0
            cpu["positions"][i] = 0
            cpu["context_lens"][i] = 1
            cpu["slot_mapping"][i] = (
                self._pad_block_id * self.block_size + i % self.block_size
            )
            block_tables[i, 0] = self._pad_block_id
        self._copy_decode_buffers(buffers, batch)

    def _capture_bucket(self, batch: int):
        device = torch.device(self.device)
        buffers = _DecodeBuffers(device, pin_memory=device.type == "cuda")
        buffers.ensure(batch, self.kv_cache.num_blocks)
        self._fill_pad_buffers(buffers, batch)
        input_ids = buffers.device_tensors["input_ids"][:batch]
        positions = buffers.device_tensors["positions"][:batch]

        warm_metadata = self._make_decode_metadata(buffers, batch)
        torch.cuda.synchronize(device)  # cpu wait for gpu to complete
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                self._decode_forward(input_ids, positions, warm_metadata)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)

        # Fresh metadata: slot_indices() must run inside capture so replay
        # recomputes block/offset tensors from the updated slot buffer.
        capture_metadata = self._make_decode_metadata(buffers, batch)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self._graph_pool), torch.inference_mode():
            logits = self._decode_forward(input_ids, positions, capture_metadata)
        if self._graph_pool is None:
            self._graph_pool = graph.pool()
        torch.cuda.synchronize(device)
        self._graph_buckets.append(
            _GraphBucket(batch, buffers, capture_metadata, graph, logits)
        )

    def capture_cuda_graphs(self):
        if self.kv_cache is None:
            raise RuntimeError("allocate the KV cache before capturing graphs")
        if torch.device(self.device).type != "cuda":
            raise RuntimeError("CUDA Graph requires a CUDA device")
        from microinfer.layer.paged_attention import flash_attn_with_kvcache
        from microinfer.layer.triton_paged_attention import HAS_TRITON

        head_dim = self.model.config.head_dim
        weight = self.model.embedding.weight
        flash_ok = flash_attn_with_kvcache is not None and self.block_size % 256 == 0
        triton_ok = (
            HAS_TRITON
            and self.block_size & (self.block_size - 1) == 0
            and head_dim & (head_dim - 1) == 0
        )
        if (
            not (flash_ok or triton_ok)
            or weight.dtype not in (torch.float16, torch.bfloat16)
            or head_dim > 256
            or head_dim % 8
        ):
            raise RuntimeError(
                "CUDA Graph decode requires FlashAttention (block_size multiple of "
                "256) or the Triton paged kernel, plus CUDA FP16/BF16 and "
                "head_dim <= 256 divisible by 8"
            )
        rotary = getattr(self.model, "rotary", None)
        if rotary is not None:
            rotary.ensure(torch.device(self.device))
        buckets = [1, 2, 4, 8]
        buckets += list(range(16, self.max_num_seqs + 1, 16))
        buckets = sorted({b for b in buckets if b <= self.max_num_seqs})
        if not buckets:
            buckets = [1]
        if buckets[-1] != self.max_num_seqs:
            buckets.append(self.max_num_seqs)
        self._graph_buckets = []
        self._graph_pool = None
        for batch in buckets:
            self._capture_bucket(batch)

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
        input_ids = torch.tensor(ids, dtype=torch.int64, device=self.device)
        position_ids = torch.tensor(positions, dtype=torch.int64, device=self.device)
        if self.debug_mode:
            self.validate_step(seqs, input_ids, position_ids, metadata, True)
        return input_ids, position_ids, metadata

    def prepare_decode(self, seqs):
        batch = len(seqs)
        width = max((len(s.block_table) for s in seqs), default=0)
        if self._decode_buffers is None:
            device = torch.device(self.device)
            self._decode_buffers = _DecodeBuffers(
                device, pin_memory=device.type == "cuda"
            )
        buffers = self._decode_buffers
        buffers.ensure(batch, width)

        cpu = buffers.cpu
        block_cpu = cpu["block_tables"]
        block_cpu[:batch].fill_(-1)
        for i, seq in enumerate(seqs):
            position = len(seq) - 1
            cpu["input_ids"][i] = seq.last_token
            cpu["positions"][i] = position
            cpu["slot_mapping"][i] = (
                seq.block_table[position // self.block_size] * self.block_size
                + position % self.block_size
            )
            cpu["context_lens"][i] = len(seq)
            block_cpu[i, : len(seq.block_table)] = torch.tensor(
                seq.block_table, dtype=torch.int32
            )

        non_blocking = buffers.pin_memory
        device_tensors = buffers.device_tensors
        for name, _ in _DECODE_SCALARS:
            device_tensors[name][:batch].copy_(
                cpu[name][:batch], non_blocking=non_blocking
            )
        block_tables = device_tensors["block_tables"][:batch]
        block_tables.copy_(block_cpu[:batch], non_blocking=non_blocking)

        slot_tensor = device_tensors["slot_mapping"][:batch]
        context_tensor = device_tensors["context_lens"][:batch]
        input_ids = device_tensors["input_ids"][:batch]
        position_ids = device_tensors["positions"][:batch]
        metadata = AttentionMetadata.build(
            is_prefill=False,
            slot_mapping=slot_tensor,
            context_lens=context_tensor,
            block_tables=block_tables,
            validate_cache_write=self.debug_mode,
        )
        if self.debug_mode:
            self.validate_step(seqs, input_ids, position_ids, metadata, False)
        return input_ids, position_ids, metadata

    def validate_step(self, seqs, input_ids, positions, metadata, is_prefill):
        """Validate scheduler/runner invariants; only called in debug_mode."""
        if metadata.is_prefill is not is_prefill:
            raise ValueError("metadata mode does not match scheduled work")
        if input_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError("input_ids and positions must be 1-D")
        if input_ids.numel() != positions.numel() or input_ids.numel() == 0:
            raise ValueError("input_ids/positions must be nonempty and equal length")
        if not seqs:
            raise ValueError("a scheduled batch must contain requests")

        block_size = self.block_size
        num_blocks = None if self.kv_cache is None else self.kv_cache.num_blocks
        slots = metadata.slot_mapping
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
        model_config = getattr(self.model, "config", None)
        max_pos = None if model_config is None else model_config.max_pos_embed
        if max_pos is not None and any(p >= max_pos for p in position_list):
            raise ValueError("position exceeds model max_pos_embed")
        token_list = [int(t) for t in input_ids.tolist()]
        if any(t < 0 for t in token_list):
            raise ValueError("input_ids must be nonnegative")
        vocab = None if model_config is None else model_config.vocab_size
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
            cu_q, cu_k = metadata.cu_seqlens_q, metadata.cu_seqlens_k
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
            if metadata.max_seqlen_q != max(q_lens):
                raise ValueError("max_seqlen_q does not match cu_seqlens_q")
            if metadata.max_seqlen_k != max(k_lens):
                raise ValueError("max_seqlen_k does not match cu_seqlens_k")
            if (
                any(seq.num_cached_tokens for seq in seqs)
                and metadata.block_tables is None
            ):
                raise ValueError("cached prefixes require block_tables")
        else:
            if len(seqs) != input_ids.numel():
                raise ValueError("decode needs exactly one token per request")
            if metadata.context_lens is None or metadata.context_lens.numel() != len(
                seqs
            ):
                raise ValueError("decode requires one context length per request")
            if metadata.block_tables is None:
                raise ValueError("decode requires block_tables")
            lens = [int(v) for v in metadata.context_lens.tolist()]
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

        if metadata.block_tables is not None:
            if metadata.block_tables.ndim != 2 or metadata.block_tables.shape[0] != len(
                seqs
            ):
                raise ValueError("block_tables must be [num_requests, max_blocks]")
            table = metadata.block_tables.tolist()
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

    def run_model(self, input_ids, positions, is_prefill, metadata):
        hidden = self._forward_for(is_prefill)(input_ids, positions, metadata)
        return self.model.compute_logits(hidden, metadata)

    def _select_graph_bucket(self, batch: int):
        for bucket in self._graph_buckets:
            if bucket.batch >= batch:
                return bucket
        return None

    def _fill_graph_bucket(self, bucket: _GraphBucket, seqs):
        batch = len(seqs)
        capacity = bucket.batch
        assert self._pad_block_id is not None
        cpu = bucket.buffers.cpu
        block_tables = cpu["block_tables"]
        block_tables.fill_(-1)

        for i, seq in enumerate(seqs):
            position = len(seq) - 1
            cpu["input_ids"][i] = seq.last_token
            cpu["positions"][i] = position
            cpu["context_lens"][i] = len(seq)
            cpu["slot_mapping"][i] = (
                seq.block_table[position // self.block_size] * self.block_size
                + position % self.block_size
            )
            block_tables[i, : len(seq.block_table)] = torch.tensor(
                seq.block_table, dtype=torch.int32
            )

        for i in range(batch, capacity):
            cpu["input_ids"][i] = 0
            cpu["positions"][i] = 0
            cpu["context_lens"][i] = 1
            cpu["slot_mapping"][i] = (
                self._pad_block_id * self.block_size + i % self.block_size
            )
            block_tables[i, 0] = self._pad_block_id

        self._copy_decode_buffers(bucket.buffers, capacity)

    def _run_decode(self, seqs):
        if self.use_cuda_graph:
            bucket = self._select_graph_bucket(len(seqs))
            if bucket is not None:
                self._fill_graph_bucket(bucket, seqs)
                if self.debug_mode:
                    tensors = bucket.buffers.device_tensors
                    batch = len(seqs)
                    self.validate_step(
                        seqs,
                        tensors["input_ids"][:batch],
                        tensors["positions"][:batch],
                        AttentionMetadata.build(
                            is_prefill=False,
                            slot_mapping=tensors["slot_mapping"][:batch],
                            context_lens=tensors["context_lens"][:batch],
                            block_tables=tensors["block_tables"][:batch],
                            validate_cache_write=False,
                        ),
                        False,
                    )
                bucket.graph.replay()
                return bucket.logits[: len(seqs)]
        ids, positions, metadata = self.prepare_decode(seqs)
        return self.run_model(ids, positions, False, metadata)

    @torch.inference_mode()
    def run(self, seqs, is_prefill):
        if not seqs:
            return []
        self.model.eval()
        if is_prefill:
            ids, positions, metadata = self.prepare_prefill(seqs)
            logits = self.run_model(ids, positions, True, metadata)
        else:
            logits = self._run_decode(seqs)
        greedy_flags = [s.greedy for s in seqs]
        if all(greedy_flags):
            tokens = logits.argmax(-1)
        elif not any(greedy_flags):
            tokens = self.sampler(logits, self.prepare_sample(seqs))
        else:
            tokens = self.sampler(logits, self.prepare_sample(seqs))
            greedy = torch.tensor(greedy_flags, device=logits.device, dtype=torch.bool)
            tokens = torch.where(greedy, logits.argmax(-1), tokens)
        return tokens.tolist()
