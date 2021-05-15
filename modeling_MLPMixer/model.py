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
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
