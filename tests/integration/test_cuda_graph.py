import pytest
import torch

from microinfer.config import QwenConfig, SampleParameter
from microinfer.engine.sequence import Sequence
from microinfer.layer.sampler import Sampler
from microinfer.model.qwen3 import QwenModel
from microinfer.runner import ModelRunner

pytestmark = [pytest.mark.cuda, pytest.mark.integration]


def _config():
    return QwenConfig(
        vocab_size=32,
        hidden_dim=16,
        MLP_inter_dim=32,
        num_q_head=2,
        num_kv_head=1,
        head_dim=8,
        num_hidden_layer=1,
        max_pos_embed=64,
        rope_base=10000.0,
    )


def _decode_seq(tokens):
    seq = Sequence(
        tokens, SampleParameter(max_tokens=1, greedy=True, ignore_eos=True)
    )
    seq.block_table = [0]
    seq.num_cached_tokens = len(tokens) - 1
    seq.num_scheduled_tokens = 1
    return seq


def test_graph_decode_matches_eager_with_padded_bucket():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    pytest.importorskip("flash_attn")
    torch.manual_seed(0)
    config = _config()
    graph_model = QwenModel(config).to("cuda", dtype=torch.bfloat16).eval()
    eager_model = QwenModel(config).to("cuda", dtype=torch.bfloat16).eval()
    eager_model.load_state_dict(graph_model.state_dict())

    graph_runner = ModelRunner(
        graph_model,
        None,
        Sampler(),
        block_size=256,
        device="cuda",
        use_cuda_graph=True,
        max_num_seqs=4,
    )
    graph_runner.allocate_kv_cache(2)
    eager_runner = ModelRunner(
        eager_model, None, Sampler(), block_size=256, device="cuda"
    )
    eager_runner.allocate_kv_cache(2)

    assert [bucket.batch for bucket in graph_runner._graph_buckets] == [1, 2, 4]
    # Three requests pad to bucket 4; context lengths differ.
    graph_seqs = [_decode_seq(tokens) for tokens in ([1], [2, 3], [4, 5, 6])]
    eager_seqs = [_decode_seq(tokens) for tokens in ([1], [2, 3], [4, 5, 6])]

    with torch.inference_mode():
        graph_tokens = graph_runner.run(graph_seqs, False)
        eager_tokens = eager_runner.run(eager_seqs, False)

    assert graph_tokens == eager_tokens
    assert len(graph_tokens) == 3


def test_graph_and_compile_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelRunner(
            None,
            None,
            None,
            block_size=4,
            use_compile=True,
            use_cuda_graph=True,
        )
