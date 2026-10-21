import torch
from torch import nn


class Sampler(nn.Module):
    def probabilities(self, logits, temperatures):
        if (
            logits.ndim != 2
            or logits.shape[1] == 0
            or not logits.dtype.is_floating_point
        ):
            raise ValueError("logits must be floating [requests, vocabulary]")
        if (
            temperatures.shape != (logits.shape[0],)
            or temperatures.device != logits.device
        ):
            raise ValueError("temperature shape/device does not match logits")
        if not torch.isfinite(temperatures).all() or (temperatures <= 0).any():
            raise ValueError("temperatures must be finite and positive")
        if (
            torch.isnan(logits).any()
            or torch.isposinf(logits).any()
            or not torch.isfinite(logits).any(-1).all()
        ):
            raise ValueError(
                "each logits row must contain finite values, without NaN/+inf"
            )
        # Center before temperature scaling to avoid positive overflow.
        work = logits.float()
        return ((work - work.amax(-1, keepdim=True)) / temperatures[:, None]).softmax(
            -1
        )

    def sample(self, logits, temperatures, *, generator=None, greedy=False):
        probabilities = self.probabilities(logits, temperatures)
        if greedy:
            return logits.argmax(-1)
        noise = torch.empty_like(probabilities).exponential_(generator=generator)
        return (probabilities / noise).argmax(-1)

    def forward(self, logits, temperatures):
        return self.sample(logits, temperatures)
