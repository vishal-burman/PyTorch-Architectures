import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
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
        self.activate = (activate is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

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

    def forward(self,):
        pass

class MobileNetV1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_size = config.in_size
        self.num_classes = config.num_classes

        self.features = nn.Sequential()
        init_block_channels = config.channels[0][0]
        self.features.add_module("init_block", ConvBlock(
            in_channels=config.in_channels,
            out_channels=init_block_channels,
            stride=2,
            ))

    def forward(self,):
        pass
