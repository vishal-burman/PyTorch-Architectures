import torch
import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, padding_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_idx)
        self.lin_1 = nn.Linear(self.embedding_size, self.hidden_size)
        self.lin_2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(1), 1)).squeeze(1)
        x = self.lin_1(x)
        x = self.lin_2(x)
        return x
