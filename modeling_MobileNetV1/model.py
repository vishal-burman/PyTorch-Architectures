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
        self.bn = nn.BatchNorm2d(num_features=out_channels,
                    eps=bn_eps,)
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
            **kwargs,
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
                bias=bias,
                )

    def forward(self, x):
        x = self.dw_conv_block(x)
        x = self.pw_conv_block(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_size = config.in_size
        self.num_classes = config.num_classes

        self.features = nn.Sequential()
        init_block_channels = config.channels[0][0]
        self.features.add_module("init_block", Conv3x3Block(
            in_channels=config.in_channels,
            out_channels=init_block_channels,
            stride=2,
            ))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(config.channels[1:]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if ((j == 0) and (i != 0)) else 1
                stage.add_module("unit{}".format(j + 1), DWSConv3x3Block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    ))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1,
            ))

        self.output = nn.Linear(
                in_features=in_channels,
                out_features=self.num_classes,
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.output(x)
        return logits
