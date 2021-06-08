import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.latent_dim, config.model_dim)
        self.act_1 = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(config.dropout)
        self.linear_2 = nn.Linear(config.model_dim, img_size)
        self.act_2 = nn.Tanh()

    def forward(self):
        pass

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

class LinearGAN(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
