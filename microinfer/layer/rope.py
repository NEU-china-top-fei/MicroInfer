"""Shared RoPE cos/sin cache. Qwen uses split-half rotation."""

import torch


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
