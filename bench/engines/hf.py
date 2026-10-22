"""Hugging Face Transformers adapter: static left-padded prefill + decode.

This is a naive KV-cache baseline, not continuous batching. It exists to show
the cost of eager PyTorch execution without paged attention or a scheduler.
"""

from __future__ import annotations

import time

import torch

from bench.engines.common import (
    peak_memory,
    reset_peak_memory,
    synchronize,
)
from bench.metrics import RequestTiming, RunResult


class _Clock:
    """CUDA events avoid per-step host synchronization on GPU."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"

    def mark(self):
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def elapsed_s(self, start, end) -> float:
        if not self.cuda:
            return end - start
        return start.elapsed_time(end) / 1000


class HFEngine:
    name = "hf"

    def __init__(
        self,
        model_path,
        *,
        device="cuda",
        dtype="bfloat16",
        block_size=None,
        num_blocks=None,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        max_model_len=None,
        attn_implementation="sdpa",
    ):
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.model = None
        self.meta = {}

    def load(self):
        import transformers
        from transformers import AutoModelForCausalLM

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=dtype,
                local_files_only=True,
                trust_remote_code=False,
                attn_implementation=self.attn_implementation,
            )
            .to(self.device)
            .eval()
        )
        self.meta = {
            "transformers": transformers.__version__,
            "model_class": type(self.model).__name__,
            "attention": getattr(
                self.model.config, "_attn_implementation", self.attn_implementation
            ),
            "static_batch": True,
            "padding": "left",
            "logits_to_keep": 1,
            "dtype": self.dtype,
        }
        return self

    def run(self, workload, *, warmup=1, runs=3, timeout_s=600):
        for _ in range(warmup):
            self._run_once(workload, timeout_s)
        return [self._run_once(workload, timeout_s) for _ in range(runs)]

    @torch.inference_mode()
    def _run_once(self, workload, timeout_s):
        prompts = [list(p) for p in workload.prompts]
        batch = len(prompts)
        max_len = max(len(p) for p in prompts)
        pad_id = 0

        input_ids = torch.full(
            (batch, max_len), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (batch, max_len), dtype=torch.long, device=self.device
        )
        for row, prompt in enumerate(prompts):
            length = len(prompt)
            input_ids[row, max_len - length :] = torch.tensor(
                prompt, dtype=torch.long, device=self.device
            )
            attention_mask[row, max_len - length :] = 1
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        clock = _Clock(self.device)
        synchronize(self.device)
        reset_peak_memory(self.device)
        wall_start = time.perf_counter()
        start_mark = clock.mark()
        marks = []

        try:
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                logits_to_keep=1,
            )
            marks.append(clock.mark())
            next_tokens = out.logits[:, -1, :].argmax(-1, keepdim=True)
            last_position = position_ids[:, -1:].clone()

            for _ in range(workload.output_len - 1):
                last_position += 1
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((batch, 1), dtype=torch.long, device=self.device),
                    ],
                    dim=1,
                )
                out = self.model(
                    next_tokens,
                    attention_mask=attention_mask,
                    position_ids=last_position,
                    past_key_values=out.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
                marks.append(clock.mark())
                next_tokens = out.logits[:, -1, :].argmax(-1, keepdim=True)
                if time.perf_counter() - wall_start > timeout_s:
                    raise TimeoutError(f"workload exceeded {timeout_s}s")
        except Exception:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise

        synchronize(self.device)
        wall_s = time.perf_counter() - wall_start
        token_times = tuple(
            clock.elapsed_s(start_mark, mark) for mark in marks
        )
        timings = tuple(
            RequestTiming(len(prompt), token_times) for prompt in prompts
        )
        alloc_mb, reserved_mb = peak_memory(self.device)
        return RunResult(
            wall_s=wall_s,
            timings=timings,
            steps=workload.output_len,
            peak_alloc_mb=alloc_mb,
            peak_reserved_mb=reserved_mb,
            extra={"static_batch": True},
        )
