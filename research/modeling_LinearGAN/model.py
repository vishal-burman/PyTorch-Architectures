import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.latent_dim, config.model_dim)
        self.act_1 = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(config.dropout)
        self.linear_2 = nn.Linear(config.model_dim, config.img_size)
        self.act_2 = nn.Tanh()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.act_2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.img_size, config.model_dim)
        self.act_1 = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(config.dropout)
        self.linear_2 = nn.Linear(config.model_dim, 1)
        self.act_2 = nn.Sigmoid()

    def forward(self):
        pass

class LinearGAN(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
