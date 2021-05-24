class MLPMixerConfig:
    def __init__(self,
            num_classes=2,
            num_blocks=4,
            image_size=256,
            patch_size=16,
            hidden_dim=32,
            tokens_mlp_dim=64,
            channels_mlp_dim=128,):

        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
