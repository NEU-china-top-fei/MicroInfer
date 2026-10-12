import torch
from torch import nn
from microinfer.config import QwenConfig
from microinfer.layer import *


class QwenModel(nn.modules):
    """
    fuse the Qwenmodel and QwenforcausualLM(frist edition at 26_10_9)
    """

    def __init__(self, config: QwenConfig):
        super.__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        # TODO
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(config) for _ in config.num_hidden_layer]
        )
        self.out_logits = nn.Linear(config.hidden_dim, config.vocab_size)
        # TODO
        self.mid_norm = FuseLayerNorm(config)

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        [batch_size,seq_len,vocab_size]->[batch_size,seq_len,hidden_dim]
        """
        hidden = self.embedding(x)
        residual = None
        for layer in self.transformer_layers:
            hidden, residual = layer(hidden, residual, position)
        return self.mid_norm(hidden)

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        [batch_size,seq_len,hidden_dim]->[batch_size,seq_len,vocab_size]
        """
        return self.out_logits(x)

    def forward_batch(self, batch, kv_cache) -> torch.Tensor:
        """M07: packed cached forward, returns logits [query_tokens, vocab_size].

        Use batch.positions for RoPE. At every layer write this step's K/V to
        batch.slot_mapping, then read paged context with the M04 backend.
        Keep the existing dense forward API for standalone layer development.
        """
        raise NotImplementedError("M07: QwenModel.forward_batch")
