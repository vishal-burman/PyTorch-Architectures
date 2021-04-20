import torch
import torch.nn as nn
import torch.nn.functional as F

class XLNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self):
        pass

class XLNetLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_type = config.attn_type
        self.same_length =config.same_length
        
        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)

    def forward(self):
        pass
