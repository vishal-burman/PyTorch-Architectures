import torch
import torch.nn as nn
import torch.nn.functional as F

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_position_embed, padding_idx=None, p_drop=0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_position_embed = max_position_embed
        self.padding_idx = padding_idx
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(self.p_drop)
    
    def forward(self):
        pass
