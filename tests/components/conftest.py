import pytest

from microinfer.config.engine import EngineConfig
from microinfer.engine.blockmanager import BlockManager
from microinfer.engine.scheduler import Scheduler


@pytest.fixture
def scheduler_factory(sequence_cls):
    def make(*, num_blocks=8, **limits):
        defaults = {
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8,
            "max_model_len": 32,
            "eos_token_id": 0,
        }
        defaults.update(limits)
        manager = BlockManager(num_blocks, sequence_cls.block_size)
        return Scheduler(EngineConfig(**defaults), manager), manager

    return make
