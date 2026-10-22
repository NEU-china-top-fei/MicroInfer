"""Shared timing/memory helpers for engine adapters."""

from __future__ import annotations

import torch

from bench.metrics import RequestTiming


def reset_prefix_cache(block_manager) -> None:
    """Clear prefix metadata without touching the free list.

    MicroInfer exposes the method; the nano-vllm reference has the same block
    fields but no helper, so fall back to the shared metadata shape.
    """
    if hasattr(block_manager, "reset_prefix_cache"):
        block_manager.reset_prefix_cache()
        return
    if block_manager.used_block_ids:
        raise RuntimeError("cannot reset prefix cache while blocks are in use")
    block_manager.hash_to_block_id.clear()
    for block in block_manager.blocks:
        block.hash = -1
        block.token_ids = []


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.max_memory_allocated(device) / 2**20,
        torch.cuda.max_memory_reserved(device) / 2**20,
    )


def collect_completion_times(
    seqs,
    times: dict[int, list[float]],
    last_counts: dict[int, int],
    elapsed_s: float,
) -> None:
    """Record one arrival per newly appended completion token.

    Call after each engine step. The current engines synchronize while converting
    sampled tokens to Python, so the step boundary is a real token arrival time.
    """
    for seq in seqs:
        count = seq.num_completion_tokens
        previous = last_counts[seq.seq_id]
        if count > previous:
            times[seq.seq_id].extend([elapsed_s] * (count - previous))
            last_counts[seq.seq_id] = count


def build_timings(workload, seqs, times: dict[int, list[float]]) -> tuple[RequestTiming, ...]:
    timings = []
    for prompt, seq in zip(workload.prompts, seqs):
        observed = len(times[seq.seq_id])
        if observed != workload.output_len:
            raise RuntimeError(
                f"request {seq.seq_id} produced {observed} tokens, "
                f"expected {workload.output_len}"
            )
        timings.append(RequestTiming(len(prompt), tuple(times[seq.seq_id])))
    return tuple(timings)
