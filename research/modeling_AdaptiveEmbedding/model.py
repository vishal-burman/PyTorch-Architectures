import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_embed,
        d_proj,
        div_val=1,
        cutoffs=[512, 1024, 2048],
    ):
        super().__init__()
        self.div_val = div_val
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [vocab_size]
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.proj_layers = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(vocab_size, d_embed))
            if d_embed != d_proj:
                self.proj_layers.append(
                    nn.Parameter(torch.FloatTensor(d_proj, d_embed))
                )
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_embed_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_embed_i))
                self.proj_layers.append(
                    nn.Parameter(torch.FloatTensor(d_proj, d_embed_i))
                )

    def forward(
        self,
        inp,  # inp ~ [batch_size, max_seq_len]
    ):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_embed != self.d_proj:
                embed = F.linear(embed, self.proj_layers[0])
        else:
            raise NotImplementedError
