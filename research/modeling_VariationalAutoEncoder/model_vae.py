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

    def reparameterize(self, z_mu, z_log_var):
        device = z_mu.device
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device)
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def encoder(self, features):
        x = self.hidden_1(features)
        x = F.leaky_relu(x, negative_slope=0.0001)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded

    def decoder(self, encoded):
        x = self.linear_3(encoded)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.linear_4(x)
        decoded = torch.sigmoid(x)
        return decoded

    def forward(self, features):
        z_mean, z_log_var, encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        return z_mean, z_log_var, encoded, decoded

