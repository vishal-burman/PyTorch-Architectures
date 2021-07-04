class MobileNetV2Config:
    def __init__(
            self,
            width_scale=1.0,
            remove_exp_conv=False,
            init_block_channels=32,
            final_block_channels=1280,
            channels=[[16], [24, 24], [32, 32, 32], [64, 64, 64, 64, 96, 96, 96], [160, 160, 160, 320]],
            in_channels=3,
            in_size=(224, 224),
            num_classes=10,
            ):
        
        self.width_scale = width_scale
        self.remove_exp_conv = remove_exp_conv
        self.init_block_channels = init_block_channels
        self.final_block_channels = final_block_channels
        self.channels = channels 
        self.in_channels = in_channels
        self.in_size = in_size
        self.num_classes = num_classes
