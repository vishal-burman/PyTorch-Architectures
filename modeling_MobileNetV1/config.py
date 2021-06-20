class MobileNetV1Config:
    def __init__(
            self,
            width_scale=1.0,
            dws_simplified=False,
            channels=[[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]],
            first_stage_stride=False,
            dw_use_bn=True,
            dw_activation=(lambda: nn.ReLU(inplace=True)),
            ):
        
        self.width_scale = width_scale
        self.dws_simplified = dws_simplified
        self.channels = channels
        self.first_stage_stride = first_stage_stride
        self.dw_use_bn = dw_use_bn
        self.dw_activation = dw_activation
