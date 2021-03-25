import torch
import torch.nn as nn
import torch.nn.functional as F

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        if self.padding_idx is not None:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_idx)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.b = nn.Parameter(torch.zeros([self.vocab_size]))

    def forward(self, x): # x ~ [batch_size, seq_len]
        x = self.embedding(x) # x ~ [batch_size, seq_len, embedding_size]
        x = x.transpose(0, 1) # x ~ [seq_len, batch_size, embedding_size]
        outputs, (_, _) = self.lstm(x) # outputs ~ [seq_len, batch_size, hidden_size]
        outputs = outputs[-1] # outputs ~ [batch_size, hidden_size]
        logits = self.W(outputs) + self.b # logits ~ [batch_size, vocab_size]
        return logits

