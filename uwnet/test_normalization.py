import torch
from .normalization import Scaler, DispatchByVariable
import pytest
from toolz import curry

approx = curry(pytest.approx, abs=1e-6)


def test_Scaler():

    x = torch.rand(10)
    mean = x.mean()
    scale = x.std()

    scaler = Scaler({'x': mean}, {'x': scale})
    y = scaler({'x': x})
    scaled = y['x']
    assert scaled.mean().item() == approx(0.0)
    assert scaled.std().item() == approx(1.0)


def test_dispatch_by_variables_membership():

    bins = [0, 1, 2, 3]
    expected = [0, 1, 3, 2, 4]
    a = torch.tensor(expected).float() - .5
    x = {'a': a.view(-1, 1)}
    model = DispatchByVariable(bins, bins, 'a', 0)
    membership = model.get_bin_membership(x)
    expected = [0, 1, 3, 2, 4]
    assert membership.tolist() == expected
