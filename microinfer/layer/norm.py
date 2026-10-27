"""Fused RMS normalization with an explicit residual accumulator."""

import torch
from torch import nn
from torch.nn import functional as F


@torch.compile(dynamic=True)
def _compiled_rms_norm(x, weight, eps):
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


@torch.compile(dynamic=True)
def _compiled_add_rms_norm(x, residual, weight, eps):
    combined = x + residual
    return F.rms_norm(combined, (combined.shape[-1],), weight, eps), combined


class FuseLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.norm_eps
        self.norm = nn.RMSNorm(config.hidden_dim, eps=self.eps)
        self.compile_ops = False

    def forward(self, x, residual=None):
        if self.compile_ops:
            if residual is None:
                return _compiled_rms_norm(x, self.norm.weight, self.eps), x
            return _compiled_add_rms_norm(x, residual, self.norm.weight, self.eps)
        combined = x if residual is None else x + residual
        normalized = F.rms_norm(
            combined, (combined.shape[-1],), self.norm.weight, self.eps
        )
        return normalized, combined
