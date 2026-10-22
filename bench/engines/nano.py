"""nano-vllm reference adapter (optional).

The reference uses a global ``Sequence.block_size`` and allocates most of the GPU
as KV cache, so run it in its own process (``bench_offline.py`` does).
"""

from __future__ import annotations

import sys
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


class NanoEngine:
    name = "nano"

    def __init__(
        self,
        model_path,
        *,
        device="cuda",
        dtype="bfloat16",
        block_size=None,
        num_blocks=None,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        max_model_len=4096,
        nano_root=None,
        enforce_eager=False,
        cache_mode="cold",
    ):
        if cache_mode not in ("cold", "warm"):
            raise ValueError("cache_mode must be cold or warm")
        self.model_path = str(model_path)
        self.cache_mode = cache_mode
        self.device = torch.device(device)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.nano_root = nano_root
        self.enforce_eager = enforce_eager
        self.llm = None
        self.meta = {}

    def load(self):
        if self.nano_root:
            sys.path.insert(0, str(self.nano_root))
        try:
            from nanovllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "nano-vllm is not importable; install it or pass --nano-root"
            ) from exc

        self._sampling_params = SamplingParams
        self.llm = LLM(
            self.model_path,
            enforce_eager=self.enforce_eager,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        block_manager = self.llm.scheduler.block_manager
        self.meta = {
            "reference": "nano-vllm",
            "enforce_eager": self.enforce_eager,
            "block_size": self.llm.scheduler.block_size,
            "num_blocks": len(block_manager.blocks),
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
            "sampling": "temperature=0.6 (reference has no greedy mode)",
            "cache_mode": self.cache_mode,
        }
        return self

    def run(self, workload, *, warmup=1, runs=3, timeout_s=600):
        for _ in range(warmup):
            self._run_once(workload, timeout_s)
        return [self._run_once(workload, timeout_s) for _ in range(runs)]

    def _run_once(self, workload, timeout_s):
        block_manager = self.llm.scheduler.block_manager
        if self.cache_mode == "cold":
            reset_prefix_cache(block_manager)
            if workload.shared_prefix:
                self._prime_prefix(workload)

        sampling = self._sampling_params(
            temperature=0.6,
            max_tokens=workload.output_len,
            ignore_eos=True,
        )

        synchronize(self.device)
        reset_peak_memory(self.device)
        start = time.perf_counter()

        for prompt in workload.prompts:
            self.llm.add_request(list(prompt), sampling)
        seqs = list(self.llm.scheduler.waiting)

        times = {seq.seq_id: [] for seq in seqs}
        last_counts = {seq.seq_id: 0 for seq in seqs}
        steps = 0
        while not self.llm.is_finished():
            self.llm.step()
            steps += 1
            collect_completion_times(
                seqs, times, last_counts, time.perf_counter() - start
            )
            if time.perf_counter() - start > timeout_s:
                raise TimeoutError(f"workload exceeded {timeout_s}s")

        synchronize(self.device)
        wall_s = time.perf_counter() - start
        timings = build_timings(workload, seqs, times)
        alloc_mb, reserved_mb = peak_memory(self.device)
        return RunResult(
            wall_s=wall_s,
            timings=timings,
            steps=steps,
            peak_alloc_mb=alloc_mb,
            peak_reserved_mb=reserved_mb,
            extra={"cuda_graph": not self.enforce_eager},
        )

    def _prime_prefix(self, workload):
        prefix = list(workload.prompts[0][: workload.shared_prefix])
        sampling = self._sampling_params(
            temperature=0.6, max_tokens=1, ignore_eos=True
        )
        self.llm.add_request(prefix, sampling)
        while not self.llm.is_finished():
            self.llm.step()
        block_manager = self.llm.scheduler.block_manager
        if block_manager.used_block_ids:
            raise RuntimeError("prefix priming left blocks in use")
        if not block_manager.hash_to_block_id:
            raise RuntimeError("prefix priming did not publish any blocks")
