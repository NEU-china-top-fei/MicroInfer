"""RMS normalization with an explicit residual accumulator."""

import torch
from torch import nn
from torch.nn import functional as F


class FuseLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.norm_eps
        self.is_fuse = config.fused_layer_norm
        self.norm = nn.RMSNorm(config.hidden_dim, eps=self.eps)

    def _reference_forward(self, x, residual=None):
        # Add in FP32, normalize before rounding back to the activation dtype.
        combined = x.float() if residual is None else x.float() + residual.float()
        normalized = combined * torch.rsqrt(
            combined.square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.norm.weight, combined.to(x.dtype)

    def fwd_fuse_tobe_compile(self, x, residual=None):
        """Backward-compatible reference path; ``forward`` uses it when unfused."""
        return self._reference_forward(x, residual)

    def forward(self, x, residual=None):
        if not self.is_fuse:
            return self._reference_forward(x, residual)
        # Fused RMSNorm (FP32 accumulation inside the kernel), with the residual
        # returned for the next layer. The add is one elementwise kernel; the
        # reference path above remains available for A/B and CPU debugging.
        combined = x if residual is None else x + residual
        normalized = F.rms_norm(
            combined, (combined.shape[-1],), self.norm.weight, self.eps
        )
        return normalized, combined
