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
        if self.use_exp_conv:
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
            )
        self.conv2 = dwconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
        )
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
        )

    def forward(
        self,
        x,
    ):
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
        self.features.add_module(
            "init_block",
            conv3x3_block(
                in_channels=config.in_channels,
                out_channels=config.init_block_channels,
                stride=2,
            ),
        )
        in_channels = config.init_block_channels
        for i, channels_per_stage in enumerate(config.channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)
                stage.add_module(
                    f"unit{j + 1}",
                    LinearBottleneck(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        expansion=expansion,
                        remove_exp_conv=config.remove_exp_conv,
                    ),
                )
                in_channels = out_channels
            self.features.add_module(f"stage{i + 1}", stage)
        self.features.add_module(
            "final_block",
            conv1x1_block(
                in_channels=in_channels,
                out_channels=config.final_block_channels,
            ),
        )
        in_channels = config.final_block_channels
        self.features.add_module(
            "final_pool",
            nn.AvgPool2d(
                kernel_size=7,
                stride=1,
            ),
        )

        self.output = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
        )

    def forward(
        self,
        pixel_values,
        labels=None,
    ):
        pixel_values = self.features(pixel_values)
        pixel_values = self.output(pixel_values)
        logits = pixel_values.view(pixel_values.size(0), -1)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        return (loss, logits)


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
    print("dwconv_block function tested!")

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
    print("dwconv3x3_block function tested!")

    # Check LinearBottleneck
    conv_func = LinearBottleneck(
        in_channels=3,
        out_channels=8,
        stride=1,
        expansion=True,
        remove_exp_conv=False,
    )
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
    print("LinearBottleneck class tested!")
