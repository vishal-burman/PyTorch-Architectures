import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, num_features, num_hidden_1):
        super().__init__()
        # Encoder
        self.linear_1 = nn.Linear(num_features, num_hidden_1)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()

        # Decoder
        self.linear_2 = nn.Linear(num_hidden_1, num_features)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()

    def forward(self, x): # x ~ [bs, c, h, w]
        x = x.view(x.size(0), -1) # x ~ [bs, c * h * w]
        encoded = self.linear_1(x) # encoded ~ [bs, num_hidden_1]
        encoded = F.leaky_relu(encoded) # encoded ~ [bs, num_hidden_1]

        logits = self.linear_2(encoded) # logits ~ [bs, c * h * w]
        decoded = torch.sigmoid(logits) # decoded ~ [bs, c * h * w]
        
        return decoded
