import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Ref: (w - k + 2*p) / s + 1 = o
        # => p = ceil[(s(o - 1) - w + k) / 2]

        ## Encoder
        # p = ceil[(2(14 - 1) - 28 + 3) / 2] = 1
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1) # 28x28x1 => 14x14x1
        # p = ceil[2(7 - 1) - 14 + 3) / 2] = 1
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1) # 14x14x4 => 7x7x8

        ## Decoder
        self.deconv_1 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=0) # 7x7x8 => 15x15x4
        self.deconv_2 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1) # 15x15x4 => 29x29x1

    def forward(self, x):
        # Encoder
        x = self.conv_1(x)
        x = F.leaky_relu(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        
        # Decoder
        x = self.deconv_1(x)
        x = F.leaky_relu(x)
        x = self.deconv_2(x)
        x = F.leaky_relu(x)
        x = x[:, :, :-1, :-1]
        x = torch.sigmoid(x)
        return x

