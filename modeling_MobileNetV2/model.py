import torch
import torch.nn as nn


def get_activation_layer(activation):
    assert activation is not None, "activation shouldn't be of None type"
    if hasattr(activation, "__call__"):
        return activation()
    else:
        raise TypeError("activation is not callable function")


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


def conv1x1_block(
    in_channels,
    out_channels,
    stride=1,
    padding=0,
    groups=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation=(lambda: nn.ReLU(inplace=True)),
):
    """
    1x1 version of standard convolution block
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
    )


def conv3x3_block(
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation=(lambda: nn.ReLU(inplace=True)),
):
    """
    3x3 version of the standard convolution block
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
    )


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        bias=False,
        use_bn=True,
        bn_eps=True,
        activation=(lambda: nn.ReLU(inplace=True)),
    ):
        super().__init__()
        self.activate = activation is not None
        self.use_bn = use_bn

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps,
            )
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(
        self,
        x,
    ):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x
