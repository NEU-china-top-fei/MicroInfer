import torch
from torch import nn
from ..model.config import QwenConfig


class FuseLayerNorm(nn.Module):
    """
    Layernorm implementation,can choose to use fuse with addition
    """

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.is_fuse = config.fused_layer_norm
        self.hidden_dim = config.hidden_dim
        self.eps = config.norm_eps
        if self.is_fuse:
            self.layer = nn.Parameter(torch.ones_like(self.hidden_dim))
        else:
            self.norm = nn.RMSNorm(self.hidden_dim, self.eps)

    @torch.compile
    def fwd_fuse_tobe_compile(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # convert to fp32
        origin_dtype = x.dtype
        x_input = x.float() + residual.float()
        residual = x

        normed = x_input * torch.rsqrt(
            x_input.pow(2).sum(dim=-1, keepdim=True) + self.eps
        ).to(origin_dtype)
        return self.layer(normed), residual

    def foward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_fuse:
            x_input = x + residual if residual is not None else x
            return self.norm(x_input), x
        else:
            return self.fwd_fuse_tobe_compile(x, residual)
