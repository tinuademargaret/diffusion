"""
Customized nn modules
"""

import math
import torch
import torch.nn as nn

import torch.nn.functional as F


class GroupNorm32(nn.GroupNorm):
    """Overide forward method of the parent class to use float type when computing with x for precision sake
    and cast x back to it's original data type"""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, length, *args):
        # store function input so we can run them again during backward pass
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # remove them from computational graph and set requires_grad to true so that their activations are stored
        ctx.input_tensors = [x.detach().requires_grad(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        # calculate gradient of input wrt their output
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,  # allows us to specify iputs that were not used when calculating the output
        )
        # clean up
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


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


def linear(*args, **kwargs):
    """Returns a linear module"""
    return nn.Linear(*args, **kwargs)


def zero_module(module):
    for param in module.parametets():
        param.detach().zero_()
    return module


def checkpoint(func, inputs, params, use_checkpoint):

    if use_checkpoint:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim / 2

    # creates frequencies that follows an exponential decay
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    # make dimensions match for matrix multiplication and then scale the frequencies by the timestep
    args = timesteps[:, None] * freqs[None]
    # concatenate sine and cosine values of the scaled timesteps
    embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # what is this dong btw?
    if embeddings % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding
