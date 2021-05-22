import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft2(x).real
        return x

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.dim = dim
        self.layer_norm = nn.LayerNorm(self.dim, eps=1e-12)

    def forward(self, x):
        x = self.fn(self.layer_norm(x)) + x

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_position_embed, padding_idx=None, p_drop=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_position_embed = max_position_embed
        self.padding_idx = padding_idx
        self.p_drop = p_drop
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(self.max_position_embed, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(self.p_drop)
    
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
