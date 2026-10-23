"""Test-only strict state_dict loader (previously microinfer.model.loader.load_weights)."""

from __future__ import annotations

import torch


@torch.no_grad()
def load_weights(model, weights, name_map=None):
    targets = model.state_dict(keep_vars=True)
    mapping, converted = name_map or {}, {}
    for source, value in weights.items():
        target = mapping.get(source, source)
        if (
            target in converted
            or target not in targets
            or value.shape != targets[target].shape
        ):
            raise ValueError(
                f"invalid checkpoint destination/shape: {source} -> {target}"
            )
        converted[target] = value.to(
            device=targets[target].device, dtype=targets[target].dtype
        )
    if set(converted) != set(targets):
        raise ValueError("checkpoint keys do not match model")
    for name, value in converted.items():
        targets[name].copy_(value)
