import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    def __init__(
            self,
            channels,
            groups,
    ):
        super().__init__()
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        batch, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, channels, height, width)
        return x


class Conv3x3Block(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def forward(self,):
        pass


def conv1x1(
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        bias=False,
):

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias,
    )


class DWSConv3x3Block(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def forward(self,):
        pass


class ShuffleNet(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def forward(self,):
        pass
