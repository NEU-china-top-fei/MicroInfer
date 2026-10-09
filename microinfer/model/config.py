from dataclasses import dataclass


@dataclass
class QwenConfig:
    vocab_size: int = 151936
    hidden_dim: int = 1024
    norm_eps: float = 1e-6
    MLP_inter_dim: int = 3072
    num_q_head: int = 16
    num_kv_head: int = 8
    head_dim: int = 128
    fused_layer_norm: bool = True
    num_hidden_layer: int = 28
    max_pos_embed: int = 40960
