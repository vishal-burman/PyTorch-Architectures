import torch
import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        # w ~ [nx, nf]
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        # self.weight ~ [nx, nf]
        self.weight = nn.Parameter(w)
        # self.bias ~ [1, nf]
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        # size_out ~ [batch_size, max_len, nf]
        size_out = x.size()[:-1] + (self.nf,)
        # x ~ [batch_size * max_len, nf]
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        # x ~ [batch_size, max_len, nf]
        x = x.view(*size_out)
        return x
