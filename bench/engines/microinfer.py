"""MicroInfer adapter: drives Scheduler/LLMEngine directly for per-token timings."""

from __future__ import annotations

import time

import torch

from bench.engines.common import (
    build_timings,
    collect_completion_times,
    peak_memory,
    reset_peak_memory,
    reset_prefix_cache,
    synchronize,
)
from bench.metrics import RunResult


class MicroInferEngine:
    name = "microinfer"

    def __init__(
        self,
        model_path,
        *,
        device="cuda",
        dtype="bfloat16",
        block_size=256,
        num_blocks=64,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        max_num_prefill_tokens=512,
        max_model_len=4096,
        cache_mode="cold",
        debug_mode=False,
        use_compile=False,
        fuse_weights=True,
        use_cuda_graph=False,
        mixed_batch=True,
    ):
        if cache_mode not in ("cold", "warm"):
            raise ValueError("cache_mode must be cold or warm")
        if type(debug_mode) is not bool or type(use_compile) is not bool:
            raise ValueError("debug_mode and use_compile must be booleans")
        self.model_path = str(model_path)
        self.debug_mode = debug_mode
        self.use_compile = use_compile
        self.fuse_weights = fuse_weights
        self.use_cuda_graph = use_cuda_graph
        self.cache_mode = cache_mode
        self.device = torch.device(device)
        self.dtype = dtype
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_prefill_tokens = max_num_prefill_tokens
        self.max_model_len = max_model_len
        self.mixed_batch = mixed_batch
        self.llm = None
        self.meta = {}

    def load(self):
        from microinfer import LLM
        from microinfer.config.engine import EngineConfig

        engine_config = EngineConfig(
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_prefill_tokens=self.max_num_prefill_tokens,
            max_model_len=self.max_model_len,
            mixed_batch=self.mixed_batch,
        )
        self.llm = LLM.from_pretrained(
            self.model_path,
            device=str(self.device),
            dtype=self.dtype,
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            engine_config=engine_config,
            debug_mode=self.debug_mode,
            use_compile=self.use_compile,
            fuse_weights=self.fuse_weights,
            use_cuda_graph=self.use_cuda_graph,
        )
        cache = self.llm.engine.model_runner.kv_cache
        from microinfer.layer.paged_attention import (
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
        )
        from microinfer.layer.triton_paged_attention import HAS_TRITON

        head_dim = self.llm.engine.model_runner.model.config.head_dim
        flash_paged = flash_attn_with_kvcache is not None and self.block_size % 256 == 0
        triton_paged = (
            HAS_TRITON
            and self.block_size & (self.block_size - 1) == 0
            and head_dim & (head_dim - 1) == 0
        )
        self.meta = {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "kv_cache_mb": round(cache.num_bytes / 2**20, 2),
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_prefill_tokens": self.max_num_prefill_tokens,
            "mixed_batch": self.mixed_batch,
            "max_model_len": self.max_model_len,
            "flash_attn_paged": flash_paged,
            "triton_paged_decode": triton_paged and not flash_paged,
            "decode_backend": (
                "flash" if flash_paged else ("triton" if triton_paged else "reference")
            ),
            "prefill_backend": (
                "flash-paged"
                if flash_paged
                else ("gather-varlen" if flash_attn_varlen_func else "reference")
            ),
            "cache_mode": self.cache_mode,
            "debug_mode": self.debug_mode,
            "use_compile": self.use_compile,
            "fuse_weights": self.fuse_weights,
            "use_cuda_graph": self.use_cuda_graph,
        }
        return self

    def run(self, workload, *, warmup=1, runs=3, timeout_s=600):
        for _ in range(warmup):
            self._run_once(workload, timeout_s)
        return [self._run_once(workload, timeout_s) for _ in range(runs)]

    def _run_once(self, workload, timeout_s):
        from microinfer import SampleParameter

        engine = self.llm.engine
        if not engine.is_finished():
            raise RuntimeError("engine has pending requests before a benchmark run")

        # Cold mode makes every measured batch independent: drop entries left by
        # warmup, then prime the shared prefix so cross-request reuse is still
        # exercised. Warm mode measures steady-state repeated workloads instead.
        block_manager = engine.scheduler.block_manager
        if self.cache_mode == "cold":
            reset_prefix_cache(block_manager)
            if workload.shared_prefix:
                self._prime_prefix(workload, block_manager)

        seqs = [
            self.llm.input_processor.process(
                list(prompt),
                SampleParameter(
                    temperature=1.0,
                    max_tokens=workload.output_len,
                    ignore_eos=True,
                    greedy=True,
                ),
            )
            for prompt in workload.prompts
        ]

        hits_before = block_manager.prefix_cache_hit_tokens
        peak_used_blocks = 0

        synchronize(self.device)
        reset_peak_memory(self.device)
        start = time.perf_counter()

        try:
            for seq in seqs:
                engine.add_request(seq)

            times = {seq.seq_id: [] for seq in seqs}
            last_counts = {seq.seq_id: 0 for seq in seqs}
            steps = 0
            while not engine.is_finished():
                engine.step()
                steps += 1
                peak_used_blocks = max(
                    peak_used_blocks, len(block_manager.used_block_ids)
                )
                collect_completion_times(
                    seqs, times, last_counts, time.perf_counter() - start
                )
                if time.perf_counter() - start > timeout_s:
                    raise TimeoutError(f"workload exceeded {timeout_s}s")
        except Exception:
            for seq in seqs:
                engine.abort(seq.seq_id)
            raise

        synchronize(self.device)
        wall_s = time.perf_counter() - start
        timings = build_timings(workload, seqs, times)
        alloc_mb, reserved_mb = peak_memory(self.device)
        total_blocks = len(block_manager.blocks)
        return RunResult(
            wall_s=wall_s,
            timings=timings,
            steps=steps,
            peak_alloc_mb=alloc_mb,
            peak_reserved_mb=reserved_mb,
            extra={
                "finished": engine.is_finished(),
                "kv_used_blocks_peak": peak_used_blocks,
                "kv_utilization_peak": peak_used_blocks / total_blocks,
                "prefix_cache_hit_tokens": (
                    block_manager.prefix_cache_hit_tokens - hits_before
                ),
            },
        )

    def _prime_prefix(self, workload, block_manager):
        """Publish full prefix blocks, then free them without clearing hashes."""
        from microinfer import SampleParameter

        prefix = list(workload.prompts[0][: workload.shared_prefix])
        engine = self.llm.engine
        seq = self.llm.input_processor.process(
            prefix,
            SampleParameter(
                temperature=1.0, max_tokens=1, ignore_eos=True, greedy=True
            ),
        )
        engine.add_request(seq)
        while not engine.is_finished():
            engine.step()
        if block_manager.used_block_ids:
            raise RuntimeError("prefix priming left blocks in use")
        if not block_manager.hash_to_block_id:
            raise RuntimeError("prefix priming did not publish any blocks")
