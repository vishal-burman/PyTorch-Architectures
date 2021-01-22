import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

class NiN(nn.Module):
    def __init__(self, num_classes, is_memory_efficient=False):
        super().__init__()
        self.num_classes = num_classes
        self.is_memory_efficient = is_memory_efficient
        self.classifier = nn.Sequential(
                # Ref --> floor(H_in + 2*padding - dilation(kernel-1) -1)/stride + 1

                # (32 + 2 * 2 -(5 - 1) - 1)/1 + 1 --> 32x32
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                # (32 + 2 * 0 -(1 - 1) - 1)/1 + 1 --> 32x32
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                # (32 + 2 * 0 -(1 - 1) - 1)/1 + 1 --> 32x32
                nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                # (32 + 2 * 1 -(3 - 1) - 1)/2 + 1 --> 16x16
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                # (16 + 2 * 2 -(5 - 1) - 1)/1 + 1 --> 16x16
                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                # (16 + 2 * 0 -(1 - 1) - 1)/1 + 1 --> 16x16
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                # (16 + 2 * 0 -(1 - 1) - 1)/1 + 1 --> 16x16
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                # (16 + 2 * 1 -3)/2 + 1 --> 8x8
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                # (8 + 2 * 1 -(3 - 1) - 1)/1 + 1 --> 8x8
                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # (8 + 2 * 0 -(1 - 1) - 1)/1 + 1 --> 8x8
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                # (8 + 2 * 0 -(1 - 1) - 1)/1 + 1 --> 8x8
                nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                # (8 + 2 * 0 -8)/1 + 1 --> 1x1 
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        if self.is_memory_efficient:
            x = cp.checkpoint_sequential(self.classifier, segments=1, input=x)
        else:
            x = self.classifier(x)
        logits = x.view(x.size(0), self.num_classes)
        probas = F.softmax(logits, dim=1)
        return logits, probas
