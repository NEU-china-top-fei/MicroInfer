"""No model downloads: run the complete engine with a tiny random Qwen3 model."""

import torch

from microinfer import LLM, QwenConfig, SampleParameter
from microinfer.model.qwen3 import QwenModel


class Tokenizer:
    eos_token_id = None

    def decode(self, tokens, *, skip_special_tokens=True):
        return " ".join(map(str, tokens))


def main():
    torch.manual_seed(7)
    config = QwenConfig(
        vocab_size=32,
        hidden_dim=16,
        MLP_inter_dim=32,
        num_hidden_layer=2,
        num_q_head=4,
        num_kv_head=2,
        head_dim=4,
        max_pos_embed=64,
    )
    llm = LLM.from_model(QwenModel(config), Tokenizer(), block_size=4, num_blocks=16)
    print(
        llm.generate(
            [[1, 2, 3], [1, 2, 7]],
            SampleParameter(max_tokens=4, greedy=True),
            use_tqdm=False,
        )
    )


if __name__ == "__main__":
    main()
