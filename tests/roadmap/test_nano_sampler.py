import pytest
import torch

from microinfer.layer.sampler import Sampler

pytestmark = [pytest.mark.roadmap, pytest.mark.nano]


def test_callable_sampler_returns_one_id_per_row():
    logits = torch.tensor([[-float("inf"), 0.0], [0.0, -float("inf")]])
    ids = Sampler()(logits, torch.ones(2))
    assert ids.tolist() == [1, 0] and ids.dtype == torch.int64


def test_temperature_changes_categorical_distribution(seeded_torch):
    logits = torch.tensor([0.0, 2.0]).expand(6000, -1).clone()
    ids = Sampler()(logits, torch.full((6000,), 2.0))
    expected = torch.tensor([0.0, 1.0]).softmax(0)
    actual = torch.bincount(ids, minlength=2).float() / 6000
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0)


def test_sampler_accepts_half_precision_logits():
    ids = Sampler()(
        torch.tensor([[0.0, -float("inf")]], dtype=torch.float16), torch.ones(1)
    )
    assert ids.tolist() == [0]
