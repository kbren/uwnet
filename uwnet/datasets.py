import numpy as np
import torch
from toolz import valmap
from torch.utils.data import Dataset

import xarray as xr


def _stack_or_rename(x, **kwargs):
    for key, val in kwargs.items():
        if isinstance(val, str):
            x = x.rename({val: key})
        else:
            x = x.stack(**{key: val})
    return x


def _ds_slice_to_numpy_dict(ds):
    out = {}
    for key in ds.data_vars:
        out[key] = _to_numpy(ds[key])
    return out


def _to_numpy(x: xr.DataArray):
    dim_order = ['xbatch', 'xtime', 'xfeat']
    dims = [dim for dim in dim_order if dim in x.dims]
    return x.transpose(*dims).values


def _ds_slice_to_torch(ds):
    return valmap(lambda x: torch.from_numpy(x).detach(),
                  _ds_slice_to_numpy_dict(ds))


class XRTimeSeries(Dataset):
    """A pytorch Dataset class for time series data in xarray format

    This function assumes the data has dimensions ['time', 'z', 'y', 'x'], and
    that the axes of the data arrays are all stored in that order.

    Attributes
    ----------
    data : xr.Dataset

    Examples
    --------
    >>> ds = xr.open_dataset("in.nc")
    >>> dataset = XRTimeSeries(ds)
    >>> dataset[0]

"""
    dims = ['time', 'z', 'x', 'y']
    batch_dims = ['y', 'x']

    def __init__(self, data):
        """Initialize XRTimeSeries.

        """
        self.data = data
        self.numpy_data = {key: data[key].values for key in data.data_vars}
        self.data_vars = set(data.data_vars)
        self.dims = {key: data[key].dims for key in data.data_vars}
        self.constants = {key for key in data.data_vars
                          if len({'x', 'y', 'time'} & set(data[key].dims)) == 0}

    def __len__(self):
        res = 1
        for dim in self.batch_dims:
            res *= len(self.data[dim])
        return res

    def __getitem__(self, i):

        # convert i to an array
        # this code should handle i = slice, list, etc
        i = np.arange(len(self))[i]

        # get coordinates using np.unravel_index
        # this code should probably be refactored
        batch_shape = [len(self.data[dim]) for dim in self.batch_dims]

        idxs = np.unravel_index(i, batch_shape)
        output_tensors = {}
        for key in self.data_vars:
            if key in self.constants:
                continue

            data_array = self.numpy_data[key]
            if 'z' in self.dims[key]:
                this_array_index = (slice(None), slice(None)) + idxs
            else:
                this_array_index = (slice(None),) + idxs
            output_tensors[key] = data_array[this_array_index].astype(np.float32)

        return output_tensors

    @property
    def time_dim(self):
        return self.dims[0][0]

    def torch_constants(self):
        return {
            key: torch.tensor(self.data[key].values, requires_grad=False)
            .float()
            for key in self.constants
        }

    @property
    def mean(self):
        """Mean of the contained variables"""
        ds = self.data.mean(['x', 'y', 'time'])
        return _ds_slice_to_torch(ds)

    @property
    def std(self):
        """Standard deviation of the contained variables"""
        ds = self.data.std(['x', 'y', 'time'])
        return _ds_slice_to_torch(ds)

    @property
    def scale(self):
        std = self.std
        return valmap(lambda x: x.max(), std)

    def timestep(self):
        time_dim = 'time'
        time = self.data[time_dim]
        dt = np.diff(time)

        all_equal = dt.std() / dt.mean() < 1e-6
        if not all_equal:
            raise ValueError("Data must be uniformly sampled in time")

        if time.units.startswith('d'):
            return dt[0] * 86400
        elif time.units.startswith('s'):
            return dt[0]
        else:
            raise ValueError(
                f"Units of time are {time.units}, but must be either seconds"
                "or days")
