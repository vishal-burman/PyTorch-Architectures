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


def dwconv_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=1,
    dilation=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation=(lambda: nn.ReLU(inplace=True)),
):

    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
    )


def dwconv3x3_block(
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    dilation=1,
    bias=False,
    bn_eps=1e-5,
    activation=(lambda: nn.ReLU(inplace=True)),
):

    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
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
        self.use_exp_conv = expansion or (not remove_exp_conv)

    def forward(
        self,
    ):
        pass


def _test():
    dummy_inputs = torch.rand(2, 3, 224, 224, requires_grad=False)

    # Check conv1x1
    conv_func = conv1x1(in_channels=3, out_channels=8)
    with torch.no_grad():
        dummy_outputs = conv_func(dummy_inputs)
    assert dummy_outputs.dim() == 4, "Shape error"
    assert dummy_outputs.size(1) == 8, "Output channel error"
    assert dummy_outputs.size(2) == dummy_inputs.size(
        2
    ), "Dimension modified with default values"
    assert dummy_outputs.size(3) == dummy_inputs.size(
        3
    ), "Dimension modified with default values"
    print("conv1x1 function tested!")

    # Check ConvBlock
    conv_func = ConvBlock(
        in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
    )
    with torch.no_grad():
        dummy_outputs = conv_func(dummy_inputs)
    assert conv_func.use_bn, "BatchNorm not activated"
    assert conv_func.activate, "Default ReLU not applied"
    assert dummy_outputs.dim() == 4, "Shape error"
    assert dummy_outputs.size(1) == 8, "Output channel error"
    assert dummy_outputs.size(2) == dummy_inputs.size(
        2
    ), "Dimension modified with default values"
    assert dummy_outputs.size(3) == dummy_inputs.size(
        3
    ), "Dimension modified with default values"
    print("ConvBlock class tested!")

    # Check conv1x1_block
    conv_func = conv1x1_block(in_channels=3, out_channels=8)
    with torch.no_grad():
        dummy_outputs = conv_func(dummy_inputs)
    assert dummy_outputs.dim() == 4, "Shape error"
    assert dummy_outputs.size(1) == 8, "Output channel error"
    assert dummy_outputs.size(2) == dummy_inputs.size(
        2
    ), "Dimension modified with default values"
    assert dummy_outputs.size(3) == dummy_inputs.size(
        3
    ), "Dimension modified with default values"
    print("conv1x1_block function tested!")

    # Check conv3x3_block
    conv_func = conv3x3_block(in_channels=3, out_channels=8)
    with torch.no_grad():
        dummy_outputs = conv_func(dummy_inputs)
    assert dummy_outputs.dim() == 4, "Shape error"
    assert dummy_outputs.size(1) == 8, "Output channel error"
    assert dummy_outputs.size(2) == dummy_inputs.size(
        2
    ), "Dimension modified with default values"
    assert dummy_outputs.size(3) == dummy_inputs.size(
        3
    ), "Dimension modified with default values"
    print("conv3x3_block function tested!")

    # Check dwconv_block
    conv_func = dwconv_block(in_channels=3, out_channels=1, kernel_size=3)
    with torch.no_grad():
        dummy_outputs = conv_func(dummy_inputs)
    assert dummy_outputs.dim() == 4, "Shape error"
    assert dummy_outputs.size(1) == 1, "Output channel error"
    assert dummy_outputs.size(2) == dummy_inputs.size(
        2
    ), "Dimension modified with default values"
    assert dummy_outputs.size(3) == dummy_inputs.size(
        3
    ), "Dimension modified with default values"
    print("dwconv_block function tested")

    # Check dwconv3x3_block
    conv_func = dwconv3x3_block(in_channels=3, out_channels=1)
    with torch.no_grad():
        dummy_outputs = conv_func(dummy_inputs)
    assert dummy_outputs.dim() == 4, "Shape error"
    assert dummy_outputs.size(1) == 1, "Output channel error"
    assert dummy_outputs.size(2) == dummy_inputs.size(
        2
    ), "Dimension modified with default values"
    assert dummy_outputs.size(3) == dummy_inputs.size(
        3
    ), "Dimension modified with default values"
    print("dwconv3x3_block function tested")
