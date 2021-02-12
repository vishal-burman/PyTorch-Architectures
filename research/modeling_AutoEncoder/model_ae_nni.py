import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1) # 28x28x1 => 28x28x4
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 28x28x4 => 14x14x4
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1) # 14x14x4 => 14x14x8
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 14x14x8 => 7x7x8

    def forward(self, x):
        # Encoder
        x = self.conv_1(x)
        x = F.leaky_relu(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        x = self.pool_2(x)
        return x

