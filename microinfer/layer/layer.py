import torch
from torch import nn
from ..model.config import QwenConfig


# RMS norm,can be configed to fuse the norm and residual addition
class QwenRMS(nn.modules):
    def __init__(self, hidden_dim: int, eps: float = 1e-6, fuse: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.fuse = fuse
        if fuse:
            self.weight = nn.Parameter(torch.ones_like(hidden_dim))
        else:
            self.norm = nn.RMSNorm(hidden_dim, eps)

    @torch.compile
    def fwd_with_residual(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origin_dtype = x.dtype
        x = x + residual
        residual = x
        high_precise_x = (
            x.float()
        )  # still on the device,float just may change its dtype

        normed = self.weight(
            high_precise_x
            * torch.rsqrt(high_precise_x.pow(2).sum(dim=1, keepdim=True) + self.eps)
        )
        return normed.to(origin_dtype), residual

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x->[bs*nh*sl,n]
        """
        if not self.fuse or residual is None:
            return self.norm(x), residual

        else:
            return self.fwd_with_residual(x, residual)


class QwenMLP(nn.modules):
    def __init__(self, config: QwenConfig):
        super.__init__(self)
        self.up_proj = nn.Linear(config.hidden_dim, config.MLP_inter_dim)
        self.down_proj = nn.Linear(config.MLP_inter_dim, config.hidden_dim)
        self.gate_proj = nn.Linear(config.hidden_dim, config.MLP_inter_dim)
        self.silu = nn.SiLU()

    def foward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.silu(self.up_proj(x) * self.gate_proj(x)))


# project:[batch_size,seq_len,hidden_dim]->[batch_size,num_head,seq_len,head_dim]
class Attention(nn.modules):
    def __init__(self, config: QwenConfig):
        super.__init__(self)
        self.q_proj = nn.Linear(config.hidden_dim, config.num_q_head * config.head_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_head * config.head_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_head * config.head_dim)
        self.num_q_head = config.num_q_head
        self.num_kv_head = config.num_kv_head
        self.q_norm = QwenRMS(config.head_dim, config.norm_eps)
        self.k_norm = QwenRMS(config.head_dim, config.norm_eps)

    def foward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden = x.shape
        x_reshaped = x.reshape(-1, hidden)
        Q = self.q_norm(
            self.q_proj(x_reshaped).reshape(batch_size, seq_len, self.num_q_head, -1)
        ).transpose(1, 2)
        K = self.k_norm(
            self.k_proj(x_reshaped).reshape(batch_size, seq_len, self.num_kv_head, -1)
        ).transpose(1, 2)
        V = (
            self.q_proj(x_reshaped)
            .reshape(batch_size, seq_len, self.num_kv_head, -1)
            .transpose(1, 2)
        )
