import pytest
import torch
from torch import nn

from microinfer.model.loader import load_weights

pytestmark = pytest.mark.roadmap


def test_explicit_name_mapping_dtype_conversion_and_no_aliasing():
    model = nn.Linear(3, 2).double()
    weight_identity = id(model.weight)
    source = {
        "hf.weight": torch.arange(6.0).reshape(2, 3),
        "hf.bias": torch.tensor([1.0, 2.0]),
    }
    expected = source["hf.weight"].double().clone()
    load_weights(model, source, {"hf.weight": "weight", "hf.bias": "bias"})
    assert id(model.weight) == weight_identity
    torch.testing.assert_close(model.weight, expected)
    source["hf.weight"].zero_()
    torch.testing.assert_close(model.weight, expected)
    torch.testing.assert_close(
        model(torch.ones(1, 3, dtype=torch.float64)),
        torch.tensor([[4.0, 14.0]], dtype=torch.float64),
    )


@pytest.mark.parametrize("problem", ["missing", "extra", "shape", "duplicate"])
def test_invalid_checkpoint_does_not_partially_modify_model(problem):
    model = nn.Linear(3, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    source = {"weight": torch.zeros(2, 3), "bias": torch.zeros(2)}
    mapping = None
    if problem == "missing":
        del source["bias"]
    elif problem == "extra":
        source["unknown"] = torch.zeros(1)
    elif problem == "shape":
        source["bias"] = torch.zeros(3)
    else:
        source["alias"] = torch.ones(2, 3)
        mapping = {"alias": "weight"}
    with pytest.raises(ValueError):
        load_weights(model, source, mapping)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
