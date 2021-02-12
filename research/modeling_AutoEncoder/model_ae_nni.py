import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1) # 28x28x1 => 28x28x4
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 28x28x4 => 14x14x4
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1) # 14x14x4 => 14x14x8
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 14x14x8 => 7x7x8

        # Decoder
        self.conv_3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1) #14x14x8 => 14x14x4
        self.conv_4 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1) #14x14x4 => 28x28x1
        
    def forward(self, x): # x ~ [bs, 1, 28, 28]
        # Encoder
        x = self.conv_1(x) # x ~ [bs, 4, 28, 28]
        x = F.leaky_relu(x) # x ~ [bs, 4, 28, 28]
        x = self.pool_1(x) # x ~ [bs, 4, 14, 14]
        x = self.conv_2(x) # x ~ [bs, 8, 14, 14]
        x = F.leaky_relu(x) # x ~ [bs, 8, 14, 14]
        x = self.pool_2(x) # x ~ [bs, 8, 7, 7]

        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='nearest') # x ~ [bs, 8, 14, 14]
        x = self.conv_3(x) # x ~ [bs, 4, 14, 14]
        x = F.leaky_relu(x) # x ~ [bs, 4, 14, 14]
        x = F.interpolate(x, scale_factor=2, mode='nearest') # x ~ [bs, 4, 28, 28]
        x = self.conv_4(x) # x ~ [bs, 1, 28, 28]
        logits = F.leaky_relu(x) # logits ~ [bs, 1, 28, 28]
        probas = torch.sigmoid(x) # probas ~ [bs, 1, 28, 28]
        return logits, probas

