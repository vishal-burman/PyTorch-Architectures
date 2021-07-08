class ShuffleNetConfig:
    def __init__(
            self,
            groups=1,
            width_scale=1.0,
            init_block_channels=24,
            layers=[4, 8, 4],
            ):

        if groups == 1:
            channels_per_layer = [144, 288, 576]
        else:
            raise NotImplementedError('Groups greater than 1 not implemented')


