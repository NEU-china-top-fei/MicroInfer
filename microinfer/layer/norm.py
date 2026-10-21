"""RMS normalization with an explicit residual accumulator."""

import torch
from torch import nn


class FuseLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.norm_eps
        self.is_fuse = config.fused_layer_norm
        self.norm = nn.RMSNorm(config.hidden_dim, eps=self.eps)

    def fwd_fuse_tobe_compile(self, x, residual=None):
        # Add in FP32, normalize before rounding back to the activation dtype.
        combined = x.float() if residual is None else x.float() + residual.float()
        normalized = combined * torch.rsqrt(
            combined.square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.norm.weight, combined.to(x.dtype)

    def forward(self, x, residual=None):
        return self.fwd_fuse_tobe_compile(x, residual)
