"""M06: local nano-vllm safetensors loading and an optional strict mapping helper."""

from collections.abc import Mapping

import torch
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """M06 nano-vllm default: copy into existing parameter storage."""
    raise NotImplementedError("M06: default_weight_loader")


def load_model(model: nn.Module, path: str):
    """M06 nano-vllm API: read local *.safetensors files.

    Dispatch packed_modules_mapping source names to (target_name, shard_id)
    and param.weight_loader(param, tensor, shard_id). Ordinary parameters use
    their weight_loader(param, tensor), or default_weight_loader.
    Strict transactional loading via load_weights is an optional extension.
    """
    raise NotImplementedError("M06: load_model")


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
