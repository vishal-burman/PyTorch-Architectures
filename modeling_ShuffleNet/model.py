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


class ShuffleUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups,
        downsample,
        ignore_group,
    ):
        super().__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        if self.downsample:
            out_channels -= in_channels

        self.compress_conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=(1 if ignore_group else groups),
        )
        self.compress_bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=groups,
        )
        self.dw_conv2 = depthwise_conv3x3(
            channels=mid_channels, stride=(2 if self.downsample else 1)
        )
        self.dw_bn2 = nn.BatchNorm2d(num_features=mid_channels)
        self.expand_conv3 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            groups=groups,
        )
        self.expand_bn3 = nn.BatchNorm2d(num_features=out_channels)
        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activ = nn.ReLU(inplace=True)

    def forward(
        self,
        x,
    ):
        identity = x
        x = self.compress_conv1(x)
        x = self.compress_bn1(x)
        x = self.activ(x)
        x = self.c_shuffle(x)
        x = self.dw_conv2(x)
        x = self.dw_bn2(x)
        x = self.expand_conv3(x)
        x = self.expand_bn3(x)
        if self.downsample:
            identity = self.avgpool(identity)
            x = torch.cat((x, identity), dim=1)
        else:
            x = x + identity
        x = self.activ(x)
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
        in_channels = config.init_block_channels
        for i, channels_per_stage in enumerate(config.channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                downsample = j == 0
                ignore_group = (i == 0) and (j == 0)
                stage.add_module(
                    f"unit{j + 1}",
                    ShuffleUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        groups=config.groups,
                        downsample=downsample,
                        ignore_group=ignore_group,
                    ),
                )
                in_channels = out_channels
            self.features.add_module(f"stage{i + 1}", stage)
        self.features.add_module(
            "final_pool",
            nn.AvgPool2d(
                kernel_size=7,
                stride=1,
            ),
        )
        self.output = nn.Linear(
            in_features=in_channels,
            out_features=config.num_classes,
        )

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        pixel_values,
        labels=None,
    ):
        pixel_values = self.features(pixel_values)
        pixel_values = pixel_values.view(pixel_values.size(0), -1)
        logits = self.output(pixel_values)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        return (loss, logits)
