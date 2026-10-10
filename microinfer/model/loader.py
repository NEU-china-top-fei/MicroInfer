"""M06 exercise: strict tensor-to-parameter loading, without network/file I/O."""

from collections.abc import Mapping

import torch
from torch import nn


def load_weights(
    model: nn.Module,
    weights: Mapping[str, torch.Tensor],
    name_map: Mapping[str, str] | None = None,
) -> None:
    """Map source names to destination state_dict names, identity otherwise.

    Reject missing/unexpected keys, duplicate destinations and shape mismatch
    before any parameter changes (ValueError). Copy with device/dtype conversion;
    retain Parameter identity. Do not alias the caller's tensor storage.
    Checkpoint shard iteration and Qwen name mapping are separate adapters.
    """
    raise NotImplementedError("M06: load_weights")
