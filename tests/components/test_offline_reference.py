"""Final acceptance gate. No downloads; explicit local artifacts are required."""

import json
import os
from pathlib import Path

import pytest
import torch

from microinfer.config import QwenConfig
from microinfer.runner import ModelRunner
from microinfer.worker.kv_cache import KVCache
from tests.support.batch import prepare_batch
from tests.support.contracts import ScheduledSequence, SchedulerOutput
from tests.support.loader import load_weights
from tests.support.model import forward_batch

pytestmark = [pytest.mark.component, pytest.mark.integration]


def test_real_checkpoint_logits_against_offline_reference():
    location = os.environ.get("MICROINFER_REFERENCE_DIR")
    if not location:
        pytest.skip(
            "Set MICROINFER_REFERENCE_DIR to converted weights/config/reference artifacts"
        )
    # An explicitly selected but incomplete directory must fail, not skip.
    directory = Path(location)
    from microinfer.model.qwen3 import QwenModel

    settings = json.loads((directory / "microinfer_config.json").read_text())
    config = QwenConfig(**settings)
    model = QwenModel(config).float().eval()
    load_weights(
        model,
        torch.load(directory / "weights.pt", map_location="cpu", weights_only=True),
    )
    reference = torch.load(
        directory / "case01.pt", map_location="cpu", weights_only=True
    )
    ids = reference["input_ids"]
    assert ids.ndim == 2 and ids.shape[0] == 1, "reference protocol requires one prompt"
    tokens = tuple(ids[0].tolist())
    assert tokens, "reference prompt must be nonempty"
    block_size = 4
    block_count = (len(tokens) + block_size - 1) // block_size
    cache = KVCache(
        config.num_hidden_layer,
        block_count,
        block_size,
        config.num_kv_head,
        config.head_dim,
    )
    batch = prepare_batch(
        SchedulerOutput(
            (ScheduledSequence(0, tokens, 0, tuple(range(block_count)), True),)
        ),
        block_size,
    )
    ModelRunner(model, cache, None, block_size=block_size)
    with torch.inference_mode():
        logits = forward_batch(model, batch)
    torch.testing.assert_close(
        logits, reference["logits"][0].float(), atol=1e-4, rtol=1e-4
    )
    assert logits[-1].argmax().item() == reference["logits"][0, -1].argmax().item()
