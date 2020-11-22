import math
import torch
import torch.nn as nn
import torch.nn.functional as F

gelu = F.gelu

def gelu_new(x):
    """
    Gaussian Error Linear Unit
    Implementation of the gelu activation function currently in Google Bert repo

    """

    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/Math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

