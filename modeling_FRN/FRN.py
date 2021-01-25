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
        pass
