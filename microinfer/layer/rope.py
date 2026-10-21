"""RoPE utilities. Qwen uses split-half rotation; adjacent pairs remain supported."""

import torch


def comp_freq_cis(max_len: int, base: float, dim: int) -> torch.Tensor:
    if dim <= 0 or dim % 2 or max_len < 0 or base <= 0:
        raise ValueError("invalid RoPE dimensions/base")
    inv = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)
    return torch.polar(torch.ones_like(angles), angles)


def apply(x, freq_cis, position=None, *, interleaved=True):
    """Rotate [..., T, heads, D] or [T, heads, D], without changing input."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    freq = freq_cis.to(x.device)
    if position is not None:
        freq = freq[position.to(device=x.device, dtype=torch.long)]
    cosine, sine = freq.real.unsqueeze(-2), freq.imag.unsqueeze(-2)
    work = x.float()
    if interleaved:
        first, second = work[..., ::2], work[..., 1::2]
        result = torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine), -1
        ).flatten(-2)
    else:
        first, second = work.chunk(2, dim=-1)
        result = torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine), dim=-1
        )
    return result.to(x.dtype)


def rotate(x, positions, base, *, interleaved=False):
    # Compute only requested positions, in FP32 even for half-precision models.
    inv = base ** (
        -torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32)
        / x.shape[-1]
    )
    angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv
    return apply(
        x, torch.polar(torch.ones_like(angles), angles), interleaved=interleaved
    )
