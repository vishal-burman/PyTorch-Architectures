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

    def forward(self, x): # x ~ [batch_size, height * width, dim]
        x = self.dense(self.dim, self.dim * self.expansion_factor)(x) # x ~ [batch_size, heigth * width, dim * expansion_factor]
        x = self.gelu(x) # x ~ [batch_size, height * width, dim * expansion_factor]
        x = self.dropout(x) # x ~ [batch_size, height * width, dim * expansion_factor]
        x = self.dense(self.dim * self.expansion_factor, self.dim)(x) # x ~ [batch_size, height * width, dim]
        x = self.dropout(x) # x ~ [batch_size, height * width, dim]
        return x

class MLPMixer(nn.Module):
    def __init__(self, image_size, patch_size, channel, dim, depth, num_classes, expansion_factor=4, p_drop=0.):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        assert self.image_size % patch_size == 0, 'Image size must be divisible by Patch size'
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.dim = dim
        self.conv_dim = (self.patch_size ** 2) * 3
        self.depth = depth
        self.num_classes = num_classes
        self.expansion_factor = expansion_factor
        self.p_drop = p_drop
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.proj = nn.Conv2d(in_channels=3, out_channels=self.conv_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.proj_to_embedding = nn.Linear(self.conv_dim, self.dim)
        self.mixer_layer_first = PreNormResidual(self.dim, FeedForwardLayer(self.num_patches, self.expansion_factor, self.p_drop, self.chan_first))
        self.mixer_layer_last = PreNormResidual(self.dim, FeedForwardLayer(self.dim, self.expansion_factor, self.p_drop, self.chan_last))
        self.layer_norm = nn.LayerNorm(self.dim)
        self.last_layer = nn.Linear(self.dim, self.num_classes)

    def embed_layer(self, img): # img ~ [batch_size, channels, height, width]
        proj = self.proj(img).flatten(2).transpose(1, 2) # proj ~ [batch_size, height * width, (patch_size**2) * 3]
        embed = self.proj_to_embedding(proj) # embed ~ [batch_size, height * width, dim]
        return embed
    
    def forward(self, img): # img ~ [batch_size, channels, height, width]
        embed = self.embed_layer(img) # embed ~ [batch_size, height * width, dim]
        for _ in range(self.depth):
            embed = self.mixer_layer_first(embed) # embed ~ [batch_size, height * width, dim]
            embed = self.mixer_layer_last(embed) # embed ~ [batch_size, height * width, dim]
        embed = self.layer_norm(embed) # embed ~ [batch_size, height * width, dim]
        embed = embed.mean(dim=1) # embed ~ [batch_size, dim]
        logits = self.last_layer(embed) # logits ~ [batch_size, num_classes]
        return logits
