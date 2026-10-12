import pytest
import torch

from microinfer.layer.sampler import Sampler

pytestmark = [pytest.mark.roadmap, pytest.mark.extension]


def test_temperature_scaling_stability_and_input_ownership():
    logits = torch.tensor([[1000.0, 1001.0, 1002.0], [3.0, 5.0, -float("inf")]])
    before = logits.clone()
    actual = Sampler().probabilities(logits, torch.tensor([1.0, 2.0]))
    expected = torch.tensor([[0.0, 1.0, 2.0], [1.5, 2.5, -float("inf")]]).softmax(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits, before)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()


def test_greedy_returns_token_ids_and_deterministic_ties():
    result = Sampler().sample(
        torch.tensor([[1.0, 3.0, 2.0], [2.0, 2.0, 1.0]]), torch.ones(2), greedy=True
    )
    assert result.tolist() == [1, 0]
    assert result.dtype == torch.int64


def test_random_sampling_reproducible_and_matches_distribution():
    logits = torch.tensor([0.2, 0.3, 0.5]).log().expand(6000, -1).clone()
    sampler = Sampler()
    a = sampler.sample(
        logits, torch.ones(6000), generator=torch.Generator().manual_seed(41)
    )
    b = sampler.sample(
        logits, torch.ones(6000), generator=torch.Generator().manual_seed(41)
    )
    assert torch.equal(a, b)
    frequencies = torch.bincount(a, minlength=3).float() / len(a)
    torch.testing.assert_close(
        frequencies, torch.tensor([0.2, 0.3, 0.5]), atol=0.025, rtol=0
    )


@pytest.mark.parametrize(
    "logits,temps",
    [
        ([[1.0, 2.0]], [0.0]),
        ([[1.0, 2.0]], [-1.0]),
        ([[1.0, 2.0]], [1.0, 2.0]),
        ([[float("nan"), 1.0]], [1.0]),
        ([[float("inf"), 1.0]], [1.0]),
        ([[-float("inf"), -float("inf")]], [1.0]),
    ],
)
def test_invalid_sampling_inputs(logits, temps):
    with pytest.raises(ValueError):
        Sampler().sample(torch.tensor(logits), torch.tensor(temps))


def test_empty_sampling_batch():
    result = Sampler().sample(torch.empty(0, 4), torch.empty(0))
    assert result.shape == (0,) and result.dtype == torch.int64
