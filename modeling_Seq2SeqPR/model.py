import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, p_drop=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        self.rnn = nn.GRU(self.emb_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.p_drop)

    def forward(self):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

class Seq2SeqPR(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

