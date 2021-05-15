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
    def __init__(self, dim, expansion_factor=4, p_drop=0.):
        super().__init__()
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.dropout = nn.Dropout(p_drop)
        self.gelu = nn.GELU()
        self.lin_1 = nn.Linear(self.dim, self.dim * self.expansion_factor)
        self.lin_2 = nn.Linear(self.dim * self.expansion_factor, self.dim)

    def forward(self):
        pass
