import torch
import torch.nn as nn


class Conv3x3Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            padding=1,
            bias=False,
            bn_eps=1e-5,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=bn_eps,
        )
        self.activ = nn.ReLU()

    def forward(self,):
        pass


class DWSConv3x3Block(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def forward(self,):
        pass


class LinearBottleneck(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def forward(self,):
        pass


class MobileNetV2(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def forward(self,):
        pass
