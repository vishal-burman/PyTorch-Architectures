import torch
import torch.nn as nn

class TransformerXLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_token = config.vocab_size
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head

    def init_mems(self, bs):
        mems = []
        params = next(self.parameters())
        for i in range(self.n_layer):
            empty = torch.zeros(self.mem_len, bsz, config.d_model, dtype=params.dtype, device=params.device)
            mems.append(empty)
        return mems

    def forward(self, input_ids):
        input_ids = input_ids.transpose(0, 1).contiguous()
        q_len, bs = input_ids.size()
        mems = self.init_mems(bs)

