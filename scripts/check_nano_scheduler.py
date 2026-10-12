"""Run the core Scheduler contracts against a local nano-vllm checkout.

Only imports and constructor wiring are adapted. Scheduler methods are unchanged.
This is a test-oracle check, never a replacement for MicroInfer's own tests.
"""

import argparse
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root", type=Path, default=project.parents[1] / "nano-vllm"
    )
    args, pytest_args = parser.parse_known_args()
    sys.path.insert(0, str(project))
    import pytest

    import microinfer.engine.scheduler as target

    source_path = args.reference_root / "nanovllm/engine/scheduler.py"
    source = source_path.read_text()
    adaptations = {
        "from nanovllm.config import Config": "from types import SimpleNamespace as Config",
        "from nanovllm.engine.sequence import Sequence, SequenceStatus": "from microinfer.engine.sequence import Sequence, SequenceStatus",
        "from nanovllm.engine.block_manager import BlockManager": "from microinfer.engine.blockmanager import BlockManager",
    }
    for old, new in adaptations.items():
        if old not in source:
            parser.error(f"Reference import changed: {old}")
        source = source.replace(old, new)
    reference = ModuleType("nano_scheduler_reference")
    # Execute the explicitly selected local reference, as with a normal import.
    exec(compile(source, str(source_path), "exec"), reference.__dict__)  # noqa: S102

    class Adapter(reference.Scheduler):
        def __init__(self, config, block_manager):
            super().__init__(
                SimpleNamespace(
                    max_num_seqs=config.max_num_seqs,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    eos=config.eos_token_id,
                    kvcache_block_size=block_manager.block_size,
                    num_kvcache_blocks=len(block_manager.blocks),
                )
            )
            self.block_manager = block_manager

    target.Scheduler = Adapter
    print(f"Reference scheduler: {source_path}")
    return pytest.main(
        [
            str(project / "tests/roadmap/test_scheduler.py"),
            "-m",
            "nano and not extension",
            "-q",
            *pytest_args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
