import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVariationalAutoEncoder(nn.Module):
    def __init__(self):
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

    def reparameterize(self):
        pass

    def encoder(self, features):
        pass

    def decoder(self):
        pass

    def forward(self, features):
        pass
