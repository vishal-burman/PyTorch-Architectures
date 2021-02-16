import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, num_features, num_hidden_1, num_latent):
        super().__init__()

        ## Encoder
        self.hidden_1 = nn.Linear(num_features, num_hidden_1)
        self.z_mean = nn.Linear(num_hidden_1, num_latent)
        self.z_log_var = nn.Linear(num_hidden_1, num_latent)

        ## Decoder
        self.linear_3 = nn.Linear(num_latent, num_hidden_1)
        self.linear_4 = nn.Linear(num_hidden_1, num_features)

    def reparameterize(self):
        pass

    def encoder(self):
        pass

    def decoder(self):
        pass

    def forward(self):
        pass

