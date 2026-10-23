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
    # Public compatibility helper; the model uses the shared RotaryCache below.
    inv = base ** (
        -torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32)
        / x.shape[-1]
    )
    angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv
    return apply(
        x, torch.polar(torch.ones_like(angles), angles), interleaved=interleaved
    )


class RotaryCache:
    """One FP32 cos/sin table shared by every decoder layer.

    The table is built lazily on the activation device so a model can be moved
    before its first forward. Entries are indexed by position, not recomputed.
    """

    def __init__(self, max_len: int, dim: int, base: float, *, interleaved=False):
        if max_len <= 0 or dim <= 0 or dim % 2 or base <= 0:
            raise ValueError("invalid RoPE cache dimensions/base")
        self.max_len = max_len
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        self._cos = None
        self._sin = None
        self._device = None

    def ensure(self, device):
        """Build the table eagerly; call before compiling the forward."""
        device = torch.device(device)
        if self._cos is not None and self._device == device:
            return
        inv = self.base ** (
            -torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
            / self.dim
        )
        angles = torch.outer(
            torch.arange(self.max_len, device=device, dtype=torch.float32), inv
        )
        self._cos = angles.cos()
        self._sin = angles.sin()
        self._device = device

    def _ensure(self, device):
        self.ensure(device)

    def apply(self, x, positions):
        if x.shape[-1] != self.dim:
            raise ValueError("RoPE cache head dimension does not match input")
        self.ensure(x.device)
        index = torch.as_tensor(positions, device=x.device, dtype=torch.long)
        cosine = self._cos.index_select(0, index).unsqueeze(-2)
        sine = self._sin.index_select(0, index).unsqueeze(-2)
        work = x.float()
        if self.interleaved:
            first, second = work[..., ::2], work[..., 1::2]
            result = torch.stack(
                (first * cosine - second * sine, first * sine + second * cosine), -1
            ).flatten(-2)
        else:
            first, second = work.chunk(2, dim=-1)
            result = torch.cat(
                (first * cosine - second * sine, second * cosine + first * sine),
                dim=-1,
            )
        return result.to(x.dtype)
