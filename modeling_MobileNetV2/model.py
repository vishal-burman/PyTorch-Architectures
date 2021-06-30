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

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class DWSConv3x3Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_eps=1e-5,
            ):
        super().__init__()
        self.dw_conv_block = Conv3x3Block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias,
                )

        self.pw_conv_block = Conv3x3Block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=bias
                )

    def forward(self, x):
        x = self.dw_conv_block(x)
        x = self.pw_conv_block(x)
        return x

class LinearBottleneck(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            expansion,
            remove_exp_conv,
            ):
        super().__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels
        self.use_exp_conv = (expansion or (not remove_exp_conv))

    def forward(self,):
        # Sample comment
        # sample comment
        pass


class MobileNetV2(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def forward(self,):
        pass
