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

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.act(x)
        return x

class LinearGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def generator_forward(self, x):
        img = self.generator(z)
        return img

    def discriminator_forward(self):
        pred = self.discriminator(img)
        return pred.view(-1)
