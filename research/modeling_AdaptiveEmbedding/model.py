import torch
import torch.nn as nn


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self,
        div_val=1,
        vocab_size,
        d_embed,
        d_proj,
    ):
        super().__init__()
        self.emb_layers = nn.ModuleList()
        self.proj_layers = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(vocab_size, d_embed))
            if d_embed != d_proj:
                self.proj_layers.append(torch.FloatTensor(d_proj, d_embed))
        else:
            pass

    def forward(
        self,
    ):
        pass
