import torch
import torch.nn as nn
import torch.nn.functional as F

class NNLM(nn.Module):
    def __init__(self, n_class, m, n_hidden, n_step=4):
        """
        Arguments:
        n_class --> Vocabulary size
        m --> hidden_dimension
        n_step --> max_length of sentences
        n_hidden --> hidden_dimension

        Returns:
        outputs --> TODO
        """
        super().__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, x): # x ~ [bs, max_length]
        X = self.C(x) # X ~ [bs, max_length, m]
        X = X.view(X.size(0), -1) # X ~ [bs, max_length * m]
        tanh = torch.tanh(self.d + self.H(X)) # tanh ~ [bs, n_hidden]
        output = self.b + self.W(X) + self.U(tanh) # output ~ [bs, n_class]
        return output
