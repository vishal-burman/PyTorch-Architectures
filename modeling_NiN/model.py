import torch
import torch.nn as nn
import torch.nn.functional as F

class NiN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(**kwargs)

    def forward(self, x):
        pass
