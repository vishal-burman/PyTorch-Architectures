import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, num_filters, filter_sizes, vocab_size, embedding_size):
        super().__init__()
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.W = nn.Embedding(self.vocab_size, self.embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self):
        pass
