import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
