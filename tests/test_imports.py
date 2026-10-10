import subprocess
import sys
from pathlib import Path


def test_engine_import_does_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from microinfer.engine.blockmanager import BlockManager; assert 'torch' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cpu_layers_import_without_flash_attention():
    # A real import blocker proves optional GPU code is not imported eagerly.
    code = """
import sys
import importlib.abc
class NoFlashAttention(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'flash_attn' or fullname.startswith('flash_attn.'):
            raise ImportError('FlashAttention is intentionally unavailable')
sys.meta_path.insert(0, NoFlashAttention())
from microinfer.layer import *
from microinfer.layer.transformer import FFN
assert 'flash_attn' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
