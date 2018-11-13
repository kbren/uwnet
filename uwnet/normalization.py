import torch
from toolz import curry, valmap
from torch import nn
from torch.autograd import Variable


def _numpy_to_variable(x):
    return Variable(torch.FloatTensor(x))


def _scale_var(scale, mean, x):
    x = x.double()
    mu = mean.double()
    sig = scale.double()

    x = x.sub(mu)
    x = x.div(sig + 1e-7)

    return x.float()


def scaler(scales, means, x):
    out = {}
    for key in x:
        if key in scales and key in means:
            out[key] = _scale_var(scales[key], means[key], x[key])
        else:
            out[key] = x[key]
    return out


def _dict_to_parameter_dict(x):
    out = {}
    for key in x:
        out[key] = nn.Parameter(x[key], requires_grad=False)
    return nn.ParameterDict(out)


class Scaler(nn.Module):
    """Torch class for normalizing data along the final dimension"""

    def __init__(self, mean=None, scale=None):
        "docstring"
        super(Scaler, self).__init__()
        if mean is None:
            mean = {}
        if scale is None:
            scale = {}
        self.mean = _dict_to_parameter_dict(mean)
        self.scale = _dict_to_parameter_dict(scale)

    def forward(self, x):
        out = {}
        for key in x:
            if key in self.scale and key in self.mean:
                out[key] = _scale_var(self.scale[key], self.mean[key], x[key])
            else:
                out[key] = x[key]
        return out


def bucketize(tensor, bucket_boundaries):
    """Equivalent to numpy.digitize

    Notes
    -----
    Torch does not have a built in equivalent yet. I found this snippet here:
    https://github.com/pytorch/pytorch/issues/7284
    """
    result = torch.zeros_like(tensor, dtype=torch.int32)
    for boundary in bucket_boundaries:
        result += (tensor > boundary).int()
    return result


class DispatchByVariable(nn.Module):
    """Dispatch

    """
    def __init__(self, bins, objs, variable, index):
        super(DispatchByVariable, self).__init__()
        self.bins = bins
        self.objs = objs
        self.variable = variable
        self.index = index

    def get_binning_variable(self, x):
        return x[self.variable][..., self.index]

    def get_bin_membership(self, x):
        y = self.get_binning_variable(x)
        return bucketize(y, self.bins)

