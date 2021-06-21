import torch
import torch.nn as nn

class Conv3x3Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            use_bn=True,
            bn_eps=1e-5,
            activation=nn.ReLU(inplace=True)
            ):
        super().__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
                )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels,
                    eps=bn_eps,)
        if self.activate:
            self.activ = activation

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
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
            dilation=1,
            bias=False,
            dw_use_bn=True,
            pw_use_bn=True,
            bn_eps=1e-5,
            dw_activation=nn.ReLU(inplace=True),
            pw_activation=nn.ReLU(inplace=True),
            ):
        super().__init__()
        self.dw_conv_block = Conv3x3Block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                use_bn=dw_use_bn,
                bn_eps=bn_eps,
                activation=dw_activation,
                )

        self.pw_conv_block = Conv3x3Block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                dilation=dilation,
                groups=1,
                bias=bias,
                use_bn=pw_use_bn,
                bn_eps=bn_eps,
                activation=pw_activation,
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
                stride = 2 if (j == 0) and ((i != 0) or config.first_stage_stride) else 1
                stage.add_module("unit{}".format(j + 1), DWSConv3x3Block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dw_use_bn=config.dw_use_bn,
                    dw_activation=config.dw_activation,
                    ))
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1,
            ))

        self.output = nn.Linear(in_features=in_channels, out_features=self.num_classes)

    def forward(self,):
        pass
