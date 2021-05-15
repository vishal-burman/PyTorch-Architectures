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
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
