import torch
import torch.nn as nn
from FRN import FilterResponseNormalization

class NiN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2,bias=False),
                FilterResponseNormalization(192),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0, bias=False),
                FilterResponseNormalization(160),
                nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0, bias=False),
                FilterResponseNormalization(96),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                )

    def forward(self, x):
        pass
