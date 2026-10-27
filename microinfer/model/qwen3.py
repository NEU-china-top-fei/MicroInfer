from torch import nn

from microinfer.layer.norm import FuseLayerNorm
from microinfer.layer.rope import RotaryCache
from microinfer.layer.transformer import TransformerLayer


class QwenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.rotary = RotaryCache(
            config.max_pos_embed,
            config.head_dim,
            config.rope_base,
            interleaved=config.rope_interleaved,
        )
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(config, self.rotary)
                for _ in range(config.num_hidden_layer)
            ]
        )
        self.mid_norm = FuseLayerNorm(config)
        self.out_logits = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.out_logits.weight = self.embedding.weight

    def fuse_weights(self):
        """Pack QKV and gate/up GEMMs after checkpoint loading."""
        for layer in self.transformer_layers:
            layer.attention.fuse_weights()
            layer.ffn.fuse_weights()

    def enable_compile_ops(self):
        """Compile RMSNorm/SwiGLU/RoPE individually (nano-vllm style)."""
        self.rotary.compile_ops = True
        for module in self.modules():
            if hasattr(module, "compile_ops"):
                module.compile_ops = True

    def forward(self, x, position, metadata=None):
        if x.ndim == 1 and metadata is None:
            raise ValueError("packed input requires AttentionMetadata")
        hidden = self.embedding(x)
        residual = None
        if metadata is None:
            for layer in self.transformer_layers:
                hidden, residual = layer(hidden, position, residual)
        else:
            for layer in self.transformer_layers:
                hidden, residual = layer(hidden, position, residual, metadata)
        hidden, _ = self.mid_norm(hidden, residual)
        return hidden

    def compute_logits(self, hidden, metadata=None):
        if (
            hidden.ndim == 2
            and metadata is not None
            and metadata.is_prefill
            and metadata.cu_seqlens_q is not None
        ):
            hidden = hidden[metadata.cu_seqlens_q[1:].long() - 1]
        return self.out_logits(hidden)
