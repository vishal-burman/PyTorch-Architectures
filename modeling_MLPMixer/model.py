import torch.nn as nn
from functools import partial

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForwardLayer(nn.Module):
    def __init__(self, dim, expansion_factor=4, p_drop=0., dense=nn.Linear):
        super().__init__()
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.dropout = nn.Dropout(p_drop)
        self.gelu = nn.GELU()
        self.dense = dense

    def forward(self, x):
        x = self.dense(self.dim, self.dim * self.expansion_factor)(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.dense(self.dim * self.expansion_factor, self.dim)(x)
        x = self.dropout(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, image_size, patch_size, channel, dim, depth, num_classes):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        assert self.image_size % patch_size == 0, 'Image size must be divisible by Patch size'
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.dim = dim
        self.conv_dim = (self.patch_size ** 2) * 3
        self.depth = depth
        self.num_classes = num_classes
        self.proj = nn.Conv2d(in_channels=3, out_channels=self.conv_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.proj_to_embedding = nn.Linear(self.conv_dim, self.dim)

    def embed_layer(self, img):
        proj = self.proj(img).flatten(2).transpose(1, 2)
        embed = self.proj_to_embedding(proj)
        return embed
    
    def forward(self):
        pass
