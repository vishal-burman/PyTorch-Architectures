class ShuffleNetConfig:
    def __init__(
        self,
        groups=1,
        width_scale=1.0,
        init_block_channels=24,
        layers=[4, 8, 4],
        in_channels=3,
        in_size=(224, 224),
        num_classes=10,
    ):

        self.groups = groups
        self.width_scale = width_scale
        self.init_block_channels = init_block_channels
        self.layers = layers
        self.in_channels = in_channels
        self.in_size = in_size
        self.num_classes = num_classes

        if self.groups == 1:
            channels_per_layer = [144, 288, 576]
        else:
            raise NotImplementedError("Groups greater than 1 not implemented")

        self.channels = [[ci] * li for (ci, li) in zip(channels_per_layer, layers)]
