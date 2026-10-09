import torch
from torch import nn
from ..model.config import QwenConfig
from norm import FuseLayerNorm
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from rope import comp_freq_cis, apply


class FFN(nn.Module):
    """
    swiLU FFN raw implementation
    """

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.up_project = nn.Linear(config.hidden_dim, config.MLP_inter_dim)
        self.down_project = nn.Linear(config.MLP_inter_dim, config.hidden_dim)
        self.gate_project = nn.Linear(config.hidden_dim, config.MLP_inter_dim)
        self.non_linear = nn.SiLU()

    def foward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_project(
            self.non_linear(self.up_project(x)) * self.gate_project(x)
        )


class Attention(nn.Module):
    """
    raw attention implementaion
    """

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.num_q_head = config.num_q_head
        self.num_kv_head = config.num_kv_head
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_dim, config.num_q_head * config.head_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_head * config.head_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_head * config.head_dim)
        self.out_proj = nn.Linear(
            config.num_kv_head * config.head_dim, config.hidden_dim
        )
        self.q_norm = FuseLayerNorm(config)
        self.k_norm = FuseLayerNorm(config)
        self.max_len = config.max_pos_embed
        self.rope_base = config.rope_base

    def foward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:

        cis = comp_freq_cis(self.max_len, self.rope_base, self.head_dim)
        batch_size, seq_len, hidden_dim = x.shape
        Q = self.q_proj(x.reshape(-1, hidden_dim)).reshape(
            -1, self.num_q_head, self.head_dim
        )
        Q = apply(Q)
        K = self.k_proj(x.reshape(-1, hidden_dim)).reshape(
            -1, self.num_kv_head, self.head_dim
        )
        K = apply(K)
        V = self.v_proj(x.reshape(-1, hidden_dim)).reshape(
            -1, self.num_kv_head, self.head_dim
        )
        # self.attn(self.q_norm(Q), self.k_norm(K), V) TODO
        return self.out_proj(
            flash_attn_varlen_func(Q, K, V, softmax_scale=self.head_dim**-0.5)
        )


class TransformerLayer(nn.Module):

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.pre_attention_norm = FuseLayerNorm(config)
        self.post_attention_norm = FuseLayerNorm(config)
        # TODO
        self.attention = Attention(config)
        # TODO
        self.ffn = FFN(config)

    def foward(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre_processed, residual = self.pre_attention_norm(x, residual)
        attened = self.attention(pre_processed, position)
        post_processed, residual = self.post_attention_norm(attened, residual)
        return self.ffn(post_processed), residual
