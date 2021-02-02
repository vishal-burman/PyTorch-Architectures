import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Ref (w - k + 2*p)/s + 1 = o
        # p = (s(o - 1) - w + k) / 2
        
        # Encoder
        # p = (1(28 - 1) - 28 + 3) / 2 = 1
        self.conv_1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1) # 28x28x1 => 28x28x4
        # p = (2(14 - 1) - 28 + 2) / 2 = 0
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 28x28x4 => 14x14x4
        # p = (1(14 - 1) - 14 + 3) / 2 = 1
        self.conv_2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1) # 14x14x4 => 14x14x8
        # p = (2(7 - 1) - 14 + 2) / 2 =  0
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 14x14x8 => 7x7x8

        # Decoder
        # Ref h_out = (h - 1)*s -2*p + (k-1) + 1
        self.deconv_1 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=0) # 7x7x8 => 15x15x4
        self.deconv_2 = nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=0) # 15x15x4 => 31x31x1


    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        x = self.pool_2(x)

        x = self.deconv_1(x)
        x = F.leaky_relu(x)
        x = self.deconv_2(x)
        x = F.leaky_relu(x)
        logits = x[:, :, 2:30, 2:30]
        probas = torch.sigmoid(logits)
        return logits, probas
