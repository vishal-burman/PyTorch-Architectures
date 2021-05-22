import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FNetConfig

config = FNetConfig()

class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.p_drop)
        self.dense_1 = nn.Linear(config.dim, config.expanded_dim)
        self.dense_2 = nn.Linear(config.expanded_dim, config.dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class FourierLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft2(x).real
        return x

class PreNormResidual(nn.Module):
    def __init__(self, config, fn):
        super().__init__()
        self.fn = fn
        self.layer_norm = nn.LayerNorm(config.dim, eps=config.eps)

    def forward(self, x):
        x = self.fn(self.layer_norm(x)) + x

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embed, config.dim)
        self.layer_norm = nn.LayerNorm(config.dim, eps=config.eps)
        self.dropout = nn.Dropout(config.p_drop)
    
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.word_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class FNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([])
        for _ in config.depth:
            layers.append(nn.ModuleList([
                PreNormResidual(config, FourierLayer()),
                PreNormResidual(config, FeedForwardLayer(config)),
                ]))

    def forward(self):
        pass
