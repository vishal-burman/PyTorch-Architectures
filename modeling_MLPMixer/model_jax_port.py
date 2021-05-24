# This is a straight port from JAX/Flax code to PyTorch code as mentioned in the paper

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        pass

    def forward(self):
        pass

class MLPMixer(nn.Module):
    def __init__(self, num_classes=2, num_blocks=4, patch_size=16, hidden_dim=32, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()
        self.num_blocks = num_blocks
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x): # x ~ [batch_size, num_channels, height, width]
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        for _ in range(self.num_blocks):
            x = MLPMixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
