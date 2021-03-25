import torch
import torch.nn as nn
import torch.nn.functional as F

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.b = nn.Parameter(torch.zeros([self.vocab_size]))

    def forward(self):
        pass

