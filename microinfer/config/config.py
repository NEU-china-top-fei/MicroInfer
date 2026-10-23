from __future__ import annotations

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
    num_hidden_layer: int = 28
    max_pos_embed: int = 40960

    rope_base: float = 1000000.0
    rope_interleaved: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf(cls, config):
        """Read a local dense Qwen3 config; unsupported variants fail explicitly."""
        if config.get("model_type") != "qwen3":
            raise ValueError("only dense Qwen3 checkpoints are supported")
        if (
            config.get("use_sliding_window")
            or config.get("hidden_act", "silu") != "silu"
        ):
            raise ValueError(
                "sliding-window attention and non-SiLU models are unsupported"
            )
        rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("scaled RoPE is not supported")
        return cls(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_size"],
            MLP_inter_dim=config["intermediate_size"],
            num_q_head=config["num_attention_heads"],
            num_kv_head=config["num_key_value_heads"],
            head_dim=config.get(
                "head_dim", config["hidden_size"] // config["num_attention_heads"]
            ),
            num_hidden_layer=config["num_hidden_layers"],
            max_pos_embed=config["max_position_embeddings"],
            norm_eps=config.get("rms_norm_eps", 1e-6),
            rope_base=rope.get("rope_theta", config.get("rope_theta", 1000000.0)),
            attention_bias=config.get("attention_bias", False),
            tie_word_embeddings=config.get("tie_word_embeddings", False),
        )
