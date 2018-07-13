import numpy as np
import pytest
import xarray as xr

from uwnet.data import XRTimeSeries


def get_obj():
    a = xr.DataArray(np.ones((3, 4, 5, 2)))
    b = xr.DataArray(np.zeros((3, 4, 5, 2)))
    c = xr.DataArray(np.zeros((3, 4, 5)))
    d = xr.DataArray(np.zeros((2,)), dims=['dim_3'])
    ds = xr.Dataset({'a': a, 'b': b, 'c': c, 'layer_mass': d})
    return ds

def test_XRTimeSeries():
    ds = get_obj()

    o = XRTimeSeries(ds, [['dim_2'], ['dim_0', 'dim_1'], ['dim_3']])
    assert len(o) == 3 * 4

    assert o[0]['a'].shape == (5, 2)
    assert o[0]['c'].shape == (5, )

    o = XRTimeSeries(ds, [['dim_1'], ['dim_0', 'dim_2'], ['dim_3']])
    assert len(o) == 3 * 5

    assert o[0]['a'].shape == (4, 2)
    assert o[0]['c'].shape == (4, )

    # try slice input
    assert o[:]['a'].shape == (15, 4, 2)


def test_XRTimeSeries_timestep():
    dt = .125
    ds = get_obj()
    ds = ds.assign_coords(dim_1=np.arange(len(ds['dim_1'])) * dt)

    # dim_1 is the time dimension here
    dataset = XRTimeSeries(ds, [['dim_1'], ['dim_0', 'dim_2'], ['dim_3']])
    assert dataset.timestep() == .125

    # non-uniformly sampled data
    time = np.arange(len(ds['dim_1']))**2
    ds = ds.assign_coords(dim_1=time)
    dataset = XRTimeSeries(ds, [['dim_1'], ['dim_0', 'dim_2'], ['dim_3']])

    with pytest.raises(ValueError):
        dataset.timestep()

def test_XRTimeSeries_constants():
    ds = get_obj()

    # dim_1 is the time dimension here
    dataset = XRTimeSeries(ds, [['dim_1'], ['dim_0', 'dim_2'], ['dim_3']])
    assert 'layer_mass' in dataset.constants()
