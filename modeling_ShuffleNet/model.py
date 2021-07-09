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
            raise ValueError("channels must be divisible by groups")
        self.groups = groups

    def forward(self, x):
        batch, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, channels, height, width)
        return x


def conv3x3(
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=False,
):

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


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


def depthwise_conv3x3(
    channels,
    stride=1,
    padding=1,
    dilation=1,
    bias=False,
):

    return nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=channels,
        bias=bias,
    )


class ShuffleInitBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(
        self,
        x,
    ):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class ShuffleNet(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.in_size = config.in_size
        self.num_classes = config.num_classes

        self.features = nn.Sequential()
        self.features.add_module(
            "init_block",
            ShuffleInitBlock(
                in_channels=config.in_channels,
                out_channels=config.init_block_channels,
            ),
        )

    def forward(
        self,
        pixel_values,
    ):
        pixel_values = self.features(pixel_values)
        return pixel_values
