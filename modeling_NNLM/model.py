import torch
import torch.nn as nn
import torch.nn.functional as F

class NNLM(nn.Module):
    def __init__(self, n_class, m, n_step, n_hidden, n_class):
        super().__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self):
        pass
