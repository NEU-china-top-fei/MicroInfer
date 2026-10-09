import torch
from torch import nn
from config import QwenConfig


class QwenModel(nn.modules):
    def __init__(self, config: QwenConfig):
        super.__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
