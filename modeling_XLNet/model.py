import torch
import torch.nn as nn
import torch.nn.functional as F

class XLNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = clamp_len
        self.n_layer = config.n_layer
        
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in config.n_layers])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids=None, attention_mask=None, perm_mask=None, target_mapping=None):
        pass

class XLNetLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_type = config.attn_type
        self.same_length =config.same_length
        
        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)

    def forward(self, input_ids=None, attention_mask=None, perm_mask=None, target_mapping=None, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, perm_mask=perm_mask, target_mapping=target_mapping)
        return transformer_outputs
