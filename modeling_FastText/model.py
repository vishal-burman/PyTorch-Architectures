import torch
import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lin_1 = nn.Linear(self.embedding_size, self.hidden_size)
        self.lin_2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self):
        pass
