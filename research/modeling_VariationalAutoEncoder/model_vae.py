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

    def reparameterize(self, z_mu, z_log_var): # z_mu ~ [bs, num_latent] || z_log_var ~ [bs, num_latent]
        device = z_mu.device # device ~ cuda:0 or cpu
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device) # eps ~ [bs, num_latent]
        z = z_mu + eps * torch.exp(z_log_var/2.) # z ~ [bs, num_latent]
        return z

    def encoder(self, features): # features ~ [bs, 784]
        x = self.hidden_1(features) # x ~ [bs, num_hidden_1]
        x = F.leaky_relu(x, negative_slope=0.0001) # x ~ [bs, num_hidden_1]
        z_mean = self.z_mean(x) # z_mean ~ [bs, num_latent]
        z_log_var = self.z_log_var(x) # z_log_var ~ [bs, num_latent]
        encoded = self.reparameterize(z_mean, z_log_var) # encoded ~ [bs, num_latent]
        return z_mean, z_log_var, encoded

    def decoder(self, encoded): # encoded ~ [bs, num_latent]
        x = self.linear_3(encoded) # x ~ [bs, num_hidden_1]
        x = F.leaky_relu(x, negative_slope=0.0001) # x ~ [bs, num_hidden_1]
        x = self.linear_4(x) # x ~ [bs, 784]
        decoded = torch.sigmoid(x) # decoded ~ [bs, 784]
        return decoded

    def forward(self, features): # features ~ [bs, 784] --> 784 if taking MNIST!!
        z_mean, z_log_var, encoded = self.encoder(features) # z_mean ~ [bs, num_latent], z_log_var ~ [bs, num_latent], encoded ~ [bs, num_latent]
        decoded = self.decoder(encoded) # decoded ~ [bs, 784]
        return z_mean, z_log_var, encoded, decoded

