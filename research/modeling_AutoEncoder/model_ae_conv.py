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
        self.conv_1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        # p = (2(14 - 1) - 28 + 2) / 2 = 0
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # p = (1(14 - 1) - 14 + 3) / 2 = 1
        self.conv_2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        # p = (2(7 - 1) - 14 + 2) / 2 = 
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self):
        pass
