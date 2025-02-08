"""
Customized nn modules
"""

import torch.nn as nn

import torch.nn.functional as F


class GroupNorm32(nn.GroupNorm):
    """Overide forward method of the parent class to use float type when computing with x for precision sake
    and cast x back to it's original data type"""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dim, *args, **kwargs):
    """
    returns a convolution layer of type dim
    """
    if dim == 1:
        return nn.Conv1d(*args, **kwargs)
    if dim == 2:
        return nn.Conv2d(*args, **kwargs)
    if dim == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError("Unsupported dim: {dim}")


def normalization(channels):
    """
    GroupNorm -> Divides the channels into groups and normalizes within each group.
    """
    return GroupNorm32(32, channels)
