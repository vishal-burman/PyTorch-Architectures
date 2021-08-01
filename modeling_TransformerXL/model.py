import torch
import torch.nn as nn


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass

    def forward(
        self,
    ):
        pass


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
