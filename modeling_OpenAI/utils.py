import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf) # w ~ [nx, nf]
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w) # self.weight ~ [nx, nf]
        self.bias = nn.Parameter(torch.zeros(nf)) # self.bias ~ [1, nf]

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,) # size_out ~ [batch_size, max_len, nf]
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight) # x ~ [batch_size * max_len, nf]
        x = x.view(*size_out) # x ~ [batch_size, max_len, nf]
        return x

def gelu_new(x):
    """
    Gaussian Error Linear Unit
    Implementation of the gelu activation function currently in Google Bert repo

    """

    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
