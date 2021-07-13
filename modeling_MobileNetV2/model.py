import torch
import torch.nn as nn


def conv1x1(
    in_channels,
    out_channels,
    stride=1,
    groups=1,
    bias=False,
):
    """
    Convolution 1x1 layer
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias,
    )
