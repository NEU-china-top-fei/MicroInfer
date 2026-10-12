"""M05: nano-vllm callable sampler, with optional probability/greedy helpers."""

import torch
from torch import nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """M05 nano-vllm entrypoint: sampler(logits, temperatures) -> int64 [N].

        Convert to FP32, scale by positive temperatures, softmax and sample.
        nano-vllm uses an exponential-race sampler; do not require its exact
        random token sequence when validating the categorical distribution.
        probabilities/sample below are optional testable MicroInfer helpers.
        """
        raise NotImplementedError("M05: Sampler.forward")

    def probabilities(
        self, logits: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        """FP32 softmax(logits / per-row temperature), no in-place changes.

        Input [S,V], temperatures [S]>0; -inf masks are allowed if each row has
        a finite entry. Reject NaN, +inf, all-masked rows and invalid shapes.
        Empty S is allowed. Invalid inputs raise ValueError.
        """
        raise NotImplementedError("M05: Sampler.probabilities")

    def sample(self, logits, temperatures, *, generator=None, greedy=False):
        """Return int64 [S]. Greedy argmax breaks ties toward the lowest ID.

        Keep current SampleParameter's positive-temperature contract; greedy is
        an explicit switch, not temperature=0. Validate both modes identically.
        """
        raise NotImplementedError("M05: Sampler.sample")
