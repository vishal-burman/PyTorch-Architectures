from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes: int, grayscale: bool = False):
        super(LeNet, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=6 * in_channels, kernel_size=5
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=6 * in_channels,
                out_channels=16 * in_channels,
                kernel_size=5,
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.Tanh(),
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.Tanh(),
            nn.Linear(84 * in_channels, num_classes),
        )

    def forward(
        self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ):
        x = self.features(pixel_values)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(logits.size(0), -1), labels.view(-1))
        return loss, logits
