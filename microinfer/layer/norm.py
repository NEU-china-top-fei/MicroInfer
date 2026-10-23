"""Fused RMS normalization with an explicit residual accumulator."""

from torch import nn
from torch.nn import functional as F


class FuseLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.norm_eps
        self.norm = nn.RMSNorm(config.hidden_dim, eps=self.eps)

    def forward(self, x, residual=None):
        combined = x if residual is None else x + residual
        normalized = F.rms_norm(
            combined, (combined.shape[-1],), self.norm.weight, self.eps
        )
        return normalized, combined
