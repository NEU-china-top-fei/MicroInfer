"""Shared RoPE cos/sin cache. Qwen uses split-half rotation."""

import torch


def _rotate(x, cosine, sine, interleaved):
    """Apply split-half or interleaved RoPE with pre-gathered cos/sin."""
    work = x.float()
    if interleaved:
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


@torch.compile(dynamic=True)
def _compiled_apply(x, positions, cos_table, sin_table, interleaved):
    index = positions.long()
    cosine = cos_table.index_select(0, index).unsqueeze(-2)
    sine = sin_table.index_select(0, index).unsqueeze(-2)
    return _rotate(x, cosine, sine, interleaved)


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
        self.compile_ops = False

    def ensure(self, device):
        """Build the table eagerly; call before compiling the forward."""
        device = torch.device(device)
        if self._cos is not None and self._device == device:
            return
        inv = self.base ** (
            -torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        )
        angles = torch.outer(
            torch.arange(self.max_len, device=device, dtype=torch.float32), inv
        )
        self._cos = angles.cos()
        self._sin = angles.sin()
        self._device = device

    def tables(self, device):
        """Return the FP32 cos/sin tables, building them on first use."""
        self.ensure(device)
        return self._cos, self._sin

    def apply(self, x, positions):
        if x.shape[-1] != self.dim:
            raise ValueError("RoPE cache head dimension does not match input")
        self.ensure(x.device)
        index = torch.as_tensor(positions, device=x.device, dtype=torch.long)
        if self.compile_ops:
            return _compiled_apply(x, index, self._cos, self._sin, self.interleaved)
        cosine = self._cos.index_select(0, index).unsqueeze(-2)
        sine = self._sin.index_select(0, index).unsqueeze(-2)
        return _rotate(x, cosine, sine, self.interleaved)
