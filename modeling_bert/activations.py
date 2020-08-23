import math

import torch
import torch.nn.functional as F


gelu = F.gelu

def gelu_new(x):
    """
    Implementation of the gelu activation function currently in Google Bert repo

    """

    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/Math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def swish(x):
    return x * torch.sigmoid(x)



ACT2FN = {
        "gelu": gelu,
        "gelu_new": gelu_new,
        "swish": swish
        }

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("Function not found!!!")
