from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from microinfer.layer.cached_attention import CachedAttention
from microinfer.layer.norm import FuseLayerNorm
from microinfer.layer.rope import RotaryCache, _rotate


@torch.compile(dynamic=True)
def _compiled_swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


@torch.compile(dynamic=True)
def _compiled_qk_norm_rope(
    q, k, q_weight, k_weight, positions, cos_table, sin_table, eps, interleaved
):
    """One compiled region for Q/K RMSNorm + RoPE along the prefill path."""
    q = F.rms_norm(q, (q.shape[-1],), q_weight, eps)
    k = F.rms_norm(k, (k.shape[-1],), k_weight, eps)
    index = positions.long()
    cosine = cos_table.index_select(0, index).unsqueeze(-2)
    sine = sin_table.index_select(0, index).unsqueeze(-2)
    return _rotate(q, cosine, sine, interleaved), _rotate(
        k, cosine, sine, interleaved
    )


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
        self.register_buffer("gate_up_weight", None, persistent=False)
        self.register_buffer("gate_up_bias", None, persistent=False)
        self.compile_ops = False

    @torch.no_grad()
    def fuse_weights(self):
        """Pack gate/up into one GEMM and make the old modules share storage.

        The projection Parameters become views into the packed tensor, so the
        checkpoint/state_dict keys stay and no duplicate weight storage remains.
        """
        if self.gate_up_weight is not None:
            return
        projections = (self.gate_project, self.up_project)
        self.gate_up_weight = torch.cat([p.weight for p in projections], dim=0)
        if any(p.bias is not None for p in projections):
            self.gate_up_bias = torch.cat(
                [
                    p.bias
                    if p.bias is not None
                    else torch.zeros(
                        p.out_features,
                        dtype=self.gate_up_weight.dtype,
                        device=self.gate_up_weight.device,
                    )
                    for p in projections
                ],
                dim=0,
            )

        # Replace originals with views into the packed buffer. The buffer is
        # non-persistent; the views keep state_dict keys and share storage.
        size = self.gate_project.out_features
        self.gate_project.weight = nn.Parameter(self.gate_up_weight[:size])
        self.up_project.weight = nn.Parameter(self.gate_up_weight[size : 2 * size])
        if self.gate_up_bias is not None:
            self.gate_project.bias = nn.Parameter(self.gate_up_bias[:size])
            self.up_project.bias = nn.Parameter(self.gate_up_bias[size : 2 * size])

    def forward(self, x):
        if self.gate_up_weight is not None:
            gate_up = F.linear(x, self.gate_up_weight, self.gate_up_bias)
            if self.compile_ops:
                return self.down_project(_compiled_swiglu(gate_up))
            gate, up = gate_up.chunk(2, dim=-1)
        else:
            gate, up = self.gate_project(x), self.up_project(x)
        return self.down_project(F.silu(gate) * up)


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
        self.rotary = rotary or RotaryCache(
            config.max_pos_embed,
            self.head_dim,
            config.rope_base,
            interleaved=config.rope_interleaved,
        )
        self.register_buffer("qkv_weight", None, persistent=False)
        self.register_buffer("qkv_bias", None, persistent=False)
        self.attn = CachedAttention()

    @torch.no_grad()
    def fuse_weights(self):
        """Pack Q/K/V into one GEMM and make the old modules share storage.

        The projection Parameters become views into the packed tensor, so the
        checkpoint/state_dict keys stay and no duplicate weight storage remains.
        """
        if self.qkv_weight is not None:
            return
        projections = (self.q_proj, self.k_proj, self.v_proj)
        self.qkv_weight = torch.cat([p.weight for p in projections], dim=0)
        if any(p.bias is not None for p in projections):
            self.qkv_bias = torch.cat(
                [
                    p.bias
                    if p.bias is not None
                    else torch.zeros(
                        p.out_features,
                        dtype=self.qkv_weight.dtype,
                        device=self.qkv_weight.device,
                    )
                    for p in projections
                ],
                dim=0,
            )

        q_size = self.num_q_head * self.head_dim
        kv_size = self.num_kv_head * self.head_dim
        self.q_proj.weight = nn.Parameter(self.qkv_weight[:q_size])
        self.k_proj.weight = nn.Parameter(self.qkv_weight[q_size : q_size + kv_size])
        self.v_proj.weight = nn.Parameter(
            self.qkv_weight[q_size + kv_size : q_size + 2 * kv_size]
        )
        if self.qkv_bias is not None:
            self.q_proj.bias = nn.Parameter(self.qkv_bias[:q_size])
            self.k_proj.bias = nn.Parameter(self.qkv_bias[q_size : q_size + kv_size])
            self.v_proj.bias = nn.Parameter(
                self.qkv_bias[q_size + kv_size : q_size + 2 * kv_size]
            )

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
        if self.qkv_weight is not None:
            qkv = F.linear(x, self.qkv_weight, self.qkv_bias)
            q, k, v = qkv.split(
                [
                    self.num_q_head * self.head_dim,
                    self.num_kv_head * self.head_dim,
                    self.num_kv_head * self.head_dim,
                ],
                dim=-1,
            )
            q = q.reshape(*shape, self.num_q_head, self.head_dim)
            k = k.reshape(*shape, self.num_kv_head, self.head_dim)
            v = v.reshape(*shape, self.num_kv_head, self.head_dim)
        else:
            q = self.q_proj(x).reshape(*shape, self.num_q_head, self.head_dim)
            k = self.k_proj(x).reshape(*shape, self.num_kv_head, self.head_dim)
            v = self.v_proj(x).reshape(*shape, self.num_kv_head, self.head_dim)
        if metadata is not None and metadata.is_prefill and self.q_norm.compile_ops:
            cos_table, sin_table = self.rotary.tables(x.device)
            q, k = _compiled_qk_norm_rope(
                q,
                k,
                self.q_norm.norm.weight,
                self.k_norm.norm.weight,
                position,
                cos_table,
                sin_table,
                self.q_norm.eps,
                self.rotary.interleaved,
            )
        else:
            q, _ = self.q_norm(q)
            k, _ = self.k_norm(k)
            q = self.rotary.apply(q, position)
            k = self.rotary.apply(k, position)
        if x.ndim == 2:
            if metadata is None:
                raise ValueError("packed attention requires AttentionMetadata")
            out = self.attn(q, k, v, metadata)
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
