import torch
import torch.nn as nn
import torch.nn.functional as F

class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        if self.padding_idx is not None:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_idx)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=True)
        self.W = nn.Linear(self.hidden_size * 2, self.vocab_size, bias=False)
        self.b = nn.Parameter(torch.ones([self.vocab_size]))

    def forward(self):
        pass

