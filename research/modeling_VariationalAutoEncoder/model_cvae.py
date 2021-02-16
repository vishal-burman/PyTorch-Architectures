import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVariationalAutoEncoder(nn.Module):
    def __init__(self, num_features, num_latent):
        super().__init__()

        ## Encoder
        self.enc_conv_1 = nn.Conv2d(1, 16, kernel_size=6, stride=2, padding=0)
        self.enc_conv_2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.enc_conv_3 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0)
        self.z_mean = nn.Linear(64*2*2, num_latent)
        self.z_log_var = nn.Linear(64*2*2, num_latent)

        ## Decoder
        self.dec_linear_1 = nn.Linear(num_latent, 64*2*2)
        self.dec_deconv_1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.dec_deconv_2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=3, padding=1)
        self.dec_deconv_3 = nn.ConvTranspose2d(16, 1, kernel_size=6, stride=3, padding=4)

    def reparameterize(self, z_mu, z_log_var): # z_mu ~ [bs, num_latent], z_log_var ~ [bs, num_latent]
        device = z_mu.device # device ~ cuda:0 or cpu
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device) # eps ~ [bs, num_latent]
        z = z_mu + eps * torch.exp(z_log_var/2.) # z ~ [bs, num_latent]
        return z

    def encoder(self, features): # features ~ [bs, 1, 28, 28]
        x = self.enc_conv_1(features) # x ~ [bs, 16, 12, 12]
        x = F.leaky_relu(x) # x ~ [bs, 16, 12, 12]

        x = self.enc_conv_2(x) # x ~ [bs, 32, 5, 5]
        x = F.leaky_relu(x) # x ~ [bs, 32, 5, 5]

        x = self.enc_conv_3(x) # x ~ [bs, 64, 2, 2]
        x = F.leaky_relu(x) # x ~ [bs, 64, 2, 2]

        z_mean = self.z_mean(x.view(-1, 64*2*2)) # z_mean ~ [bs, num_latent]
        z_log_var = self.z_log_var(x.view(-1, 64*2*2)) # z_log_var ~ [bs, num_latent]
        encoded = self.reparameterize(z_mean, z_log_var) # encoded ~ [bs, num_latent]
        return z_mean, z_log_var, encoded

    def decoder(self, encoded): # encoded ~ [bs, num_latent]
        x = self.dec_linear_1(encoded) # x ~ [bs, 64*2*2]
        x = x.view(-1, 64, 2, 2) # x ~ [bs, 64, 2, 2]

        x = self.dec_deconv_1(x) # x ~ [bs, 32, 4, 4]
        x = F.leaky_relu(x) # x ~ [bs, 32, 4, 4]

        x = self.dec_deconv_2(x) # x ~ [bs, 16, 11, 11]
        x = F.leaky_relu(x) # x ~ [bs, 16, 11, 11]

        x = self.dec_deconv_3(x) # x ~ [bs, 1, 28, 28]
        x = F.leaky_relu(x) # x ~ [bs, 1, 28, 28]

        decoded = torch.sigmoid(x) # decoded ~ [bs, 1, 28, 28]
        return decoded

    def forward(self, features):
        z_mean, z_log_var, encoded = self.encoder(features) # z_mean ~ [bs, num_latent], z_log_var ~ [bs, num_latent], encoded ~ [bs, num_latent]
        decoded = self.decoder(encoded) # decoded ~ [bs, 1, 28, 28]
        return z_mean, z_log_var, encoded, decoded
