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
            activation='relu'
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
        if activation == "relu":
            self.activ = nn.ReLU()
        elif activation == "relu6":
            self.activ = nn.ReLU6()
        elif activation is None:
            self.activ = nn.Identity()

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
            activation="relu",
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
            bn_eps=bn_eps,
            activation=activation,
        )

        self.pw_conv_block = Conv3x3Block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            bn_eps=bn_eps,
            activation=activation,
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

        if self.use_exp_conv:
            self.conv1 = Conv3x3Block(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                padding=0,
                activation="relu6",
            )
        self.conv2 = DWSConv3x3Block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            activation="relu6",
        )
        self.conv3 = Conv3x3Block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None,
        )

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV2(nn.Module):
    def __init__(
            self,
            config,
            ):
        super().__init__()
        self.in_size = config.in_size
        self.num_classes = config.num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", Conv3x3Block(
            in_channels=config.in_channels,
            out_channels=config.init_block_channels,
            stride=2,
            activation='relu6',
            ))
        in_channels = config.init_block_channels
        for i, channels_per_stage in enumerate(config.channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)
                stage.add_module("unit{}".format(j + 1), LinearBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion=expansion,
                    remove_exp_conv=config.remove_exp_conv,
                    ))
                in_channels=out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", Conv3x3Block(
            in_channels=in_channels,
            out_channels=config.final_block_channels,
            activation="relu6",
            ))
        in_channels = config.final_block_channels
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1,
            ))

        self.output = Conv3x3Block(
                in_channels=in_channels,
                out_channels=config.num_classes,
                bias=False,
                )

    def forward(self, pixel_values, labels=None):
        pixel_values = self.features(pixel_values)
        pixel_values = self.output(pixel_values)
        logits = pixel_values.view(pixel_values.size(0), -1)
        return logits
