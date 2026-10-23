from dataclasses import replace

from torch import nn
from torch.nn import functional as F

from microinfer.layer.cached_attention import CachedAttention
from microinfer.layer.norm import FuseLayerNorm
from microinfer.layer.rope import RotaryCache
from microinfer.utils.context import get_context


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_project = nn.Linear(config.hidden_dim, config.MLP_inter_dim, bias=False)
        self.gate_project = nn.Linear(
            config.hidden_dim, config.MLP_inter_dim, bias=False
        )
        self.down_project = nn.Linear(
            config.MLP_inter_dim, config.hidden_dim, bias=False
        )

    def forward(self, x):
        return self.down_project(F.silu(self.gate_project(x)) * self.up_project(x))


class Attention(nn.Module):
    def __init__(self, config, rotary=None):
        super().__init__()
        self.num_q_head, self.num_kv_head, self.head_dim = (
            config.num_q_head,
            config.num_kv_head,
            config.head_dim,
        )
        if self.num_q_head % self.num_kv_head or self.head_dim % 2:
            raise ValueError("invalid GQA head ratio or RoPE dimension")
        self.q_proj = nn.Linear(
            config.hidden_dim,
            self.num_q_head * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_dim,
            self.num_kv_head * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_dim,
            self.num_kv_head * self.head_dim,
            bias=config.attention_bias,
        )
        self.out_proj = nn.Linear(
            self.num_q_head * self.head_dim, config.hidden_dim, bias=False
        )
        head_config = replace(config, hidden_dim=self.head_dim)
        self.q_norm, self.k_norm = (
            FuseLayerNorm(head_config),
            FuseLayerNorm(head_config),
        )
        self.rope_base, self.rope_interleaved = (
            config.rope_base,
            config.rope_interleaved,
        )
        self.rotary = rotary or RotaryCache(
            config.max_pos_embed,
            self.head_dim,
            config.rope_base,
            interleaved=config.rope_interleaved,
        )
        self.attn = CachedAttention()

    def bind_cache(self, cache):
        if cache.keys.shape[2:] != (self.num_kv_head, self.head_dim):
            raise ValueError("layer cache does not match attention heads")
        self.attn.bind_cache(cache)

    def _dense_attention(self, q, k, v):
        # Standalone dense forward is stateless, even when a pool is bound.
        repeats = self.num_q_head // self.num_kv_head
        return F.scaled_dot_product_attention(
            q.transpose(-3, -2),
            k.repeat_interleave(repeats, dim=-2).transpose(-3, -2),
            v.repeat_interleave(repeats, dim=-2).transpose(-3, -2),
            is_causal=True,
        ).transpose(-3, -2)

    def forward(self, x, position, metadata=None):
        shape = x.shape[:-1]
        q = self.q_proj(x).reshape(*shape, self.num_q_head, self.head_dim)
        k = self.k_proj(x).reshape(*shape, self.num_kv_head, self.head_dim)
        v = self.v_proj(x).reshape(*shape, self.num_kv_head, self.head_dim)
        q, _ = self.q_norm(q)
        k, _ = self.k_norm(k)
        q = self.rotary.apply(q, position)
        k = self.rotary.apply(k, position)
        if x.ndim == 2 and metadata is not None:
            out = self.attn(q, k, v, metadata)
        elif x.ndim == 2:
            ctx = get_context()
            if (
                ctx.metadata is not None
                or ctx.cu_seqlens_q is not None
                or ctx.context_lens is not None
            ):
                out = self.attn(q, k, v)
            else:
                out = self._dense_attention(q, k, v)
        else:
            out = self._dense_attention(q, k, v)
        return self.out_proj(out.reshape(*shape, self.num_q_head * self.head_dim))


class TransformerLayer(nn.Module):
    def __init__(self, config, rotary=None):
        super().__init__()
        self.pre_attention_norm = FuseLayerNorm(config)
        self.post_attention_norm = FuseLayerNorm(config)
        self.attention = Attention(config, rotary)
        self.ffn = FFN(config)

    def forward(self, x, position, residual=None, metadata=None):
        hidden, residual = self.pre_attention_norm(x, residual)
        if metadata is None:
            hidden = self.attention(hidden, position)
        else:
            hidden = self.attention(hidden, position, metadata)
        hidden, residual = self.post_attention_norm(hidden, residual)
        return self.ffn(hidden), residual
