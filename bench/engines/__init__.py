"""Engine adapters. Import lazily so one missing backend does not break others."""

from __future__ import annotations

import inspect


def _filtered(cls, kwargs: dict) -> dict:
    parameters = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def build_engine(name: str, **kwargs):
    if name == "microinfer":
        from bench.engines.microinfer import MicroInferEngine

        return MicroInferEngine(**_filtered(MicroInferEngine, kwargs))
    if name == "hf":
        from bench.engines.hf import HFEngine

        return HFEngine(**_filtered(HFEngine, kwargs))
    if name == "nano":
        from bench.engines.nano import NanoEngine

        return NanoEngine(**_filtered(NanoEngine, kwargs))
    raise ValueError(f"unknown engine {name!r}; choose from microinfer, hf, nano")
