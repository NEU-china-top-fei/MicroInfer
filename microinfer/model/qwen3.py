import torch
from torch import nn

from microinfer.layer.metadata import AttentionMetadata
from microinfer.layer.norm import FuseLayerNorm
from microinfer.layer.rope import RotaryCache
from microinfer.layer.transformer import TransformerLayer
from microinfer.utils.context import Context, get_context, use_context


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

    def forward(self, x, position, metadata=None):
        if metadata is None and x.ndim == 1:
            context = get_context()
            if (
                context.metadata is not None
                or context.cu_seqlens_q is not None
                or context.context_lens is not None
            ):
                metadata = AttentionMetadata.from_context()
        hidden = self.embedding(x)
        residual = None
        for layer in self.transformer_layers:
            if metadata is None:
                hidden, residual = layer(hidden, position, residual)
            else:
                hidden, residual = layer(hidden, position, residual, metadata)
        hidden, _ = self.mid_norm(hidden, residual)
        return hidden

    def compute_logits(self, hidden, metadata=None):
        if metadata is None and hidden.ndim == 2:
            context = get_context()
            if (
                context.metadata is not None
                or context.cu_seqlens_q is not None
                or context.context_lens is not None
            ):
                metadata = AttentionMetadata.from_context()
        if (
            hidden.ndim == 2
            and metadata is not None
            and metadata.is_prefill
            and metadata.cu_seqlens_q is not None
        ):
            hidden = hidden[metadata.cu_seqlens_q[1:].long() - 1]
        return self.out_logits(hidden)

    def forward_batch(self, batch):
        """All-query logits for numerical tests; cache binding belongs to Runner."""
        lengths = batch.context_lens
        cu_k = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(torch.int32)))
        q_lengths = batch.cu_seqlens_q[1:] - batch.cu_seqlens_q[:-1]
        ctx = Context(
            True,
            batch.cu_seqlens_q,
            cu_k,
            int(q_lengths.max()) if q_lengths.numel() else 0,
            int(lengths.max()) if lengths.numel() else 0,
            batch.slot_mapping,
            None,
            batch.block_tables,
        )
        with use_context(ctx):
            return self.out_logits(self(batch.input_ids, batch.positions))
