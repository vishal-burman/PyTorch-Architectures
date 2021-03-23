import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.B = nn.Parameter(torch.ones([self.vocab_size]))

    def forward(self):
        pass
