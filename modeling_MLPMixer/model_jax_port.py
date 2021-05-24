# This is a straight port from JAX/Flax code to PyTorch code as mentioned in the paper

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, mlp_dim):
        super().__init__()
        pass

    def forward(self):
        pass

class MixerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.token_mixing = MLPBlock(config.tokens_mlp_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_blocks = config.num_blocks
        self.tokens_mlp_dim = config.tokens_mlp_dim
        self.channels_mlp_dim = config.channels_mlp_dim
        self.conv = nn.Conv2d(in_channels=3, out_channels=config.hidden_dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.layers = nn.ModuleList([])
        for _ in range(config.num_blocks):
            self.layers.append(MixerBlock(config))

    def forward(self, x): # x ~ [batch_size, num_channels, height, width]
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x
