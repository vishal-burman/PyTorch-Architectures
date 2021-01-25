import torch
import torch.nn as nn

class FilterResponseNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.register_parameter('beta', torch.nn.Parameter(torch.empty([1, num_features, 1, 1]).normal_()))
        self.register_parameter('gamma', torch.nn.Parameter(torch.empty([1, num_features, 1, 1]).normal_()))
        self.register_parameter('tau', torch.nn.Parameter(torch.empty([1, num_features, 1, 1]).normal_()))
        self.eps = torch.Tensor([eps])

    def forward(self, x):
        n, c, h, w = x.size()
        self.eps = self.eps.to(self.tau.device)
        nu2 = torch.mean(x.pow(2), (2, 3), keepdims=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)
