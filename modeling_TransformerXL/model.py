import torch
import torch.nn as nn


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self,
        n_token,
        d_embed,
        d_proj,
        cutoffs,
        div_val,
        sample_softmax=False,
    ):
        super().__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** -0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))

    def forward(
        self,
        inp,
    ):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = nn.functional.linear(embed, self.emb_projs[0])
        return embed


class TransformerXLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_token = config.vocab_size
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head

        self.word_emb = AdaptiveEmbedding(
            config.vocab_size,
            config.d_embed,
            config.d_model,
            config.cutoffs,
            div_val=config.div_val,
        )

    def init_mems(self, bs):
        mems = []
        params = next(self.parameters())
        for i in range(self.n_layer):
            empty = torch.zeros(
                self.mem_len,
                bsz,
                config.d_model,
                dtype=params.dtype,
                device=params.device,
            )
            mems.append(empty)
        return mems

    def forward(self, input_ids):
        input_ids = input_ids.transpose(0, 1).contiguous()
        q_len, bs = input_ids.size()
        mems = self.init_mems(bs)
        word_emb = self.word_emb(input_ids)
