# This is a straight port from JAX/Flax code to PyTorch code as mentioned in the paper
import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x): # x ~ [:, :, input_dim]
        x = self.dense_1(x) # x ~ [:, :, mlp_dim]
        x = self.gelu(x) # x ~ [:, :, mlp_dim]
        x = self.dense_2(x) # x ~ [:, :, input_dim]
        return x

class MixerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.token_mixing = MLPBlock(config.patch_size ** 2, config.tokens_mlp_dim)
        self.channel_mixing = MLPBlock(config.hidden_dim, config.channels_mlp_dim)

    def forward(self, x): # x ~ [batch_size, height * width // patch_size ^ 2, hidden_dim]
        y = self.layer_norm(x) # y ~ [batch_size, heigth * width // patch_size ^ 2, hidden_dim]
        y = y.transpose(1, 2) # y ~ [batch_size, hidden_dim, heigth * width // patch_size ^ 2] 
        y = self.token_mixing(y).transpose(1, 2) # y ~ [batch_size, height * width // patch_size ^ 2, hidden_dim]
        x = x + y # x ~ [batch_size, height * width // patch_size ^ 2, hidden_dim]
        y = self.layer_norm(x) # y ~ [batch_size, height * width // patch_size ^ 2, hidden_dim]
        x = x + self.channel_mixing(y) # x ~ [batch_size, height * width // patch_size ^ 2, hidden_dim]
        return x

class MLPMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_blocks = config.num_blocks
        self.tokens_mlp_dim = config.tokens_mlp_dim
        self.channels_mlp_dim = config.channels_mlp_dim
        self.conv = nn.Conv2d(in_channels=3, out_channels=config.hidden_dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.layers = nn.ModuleList([])
        for _ in range(config.num_blocks):
            self.layers.append(MixerBlock(config))
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x, labels=None): # x ~ [batch_size, num_channels, height, width]
        x = self.conv(x).flatten(2).transpose(1, 2) # x ~ [batch_size, height * width // patch_size ^ 2, hidden_dim]
        for layer in self.layers:
            x = layer(x) # x ~ [batch_size, height * width // patch_size ^ 2, hidden_dim]
        x = self.layer_norm(x) # x ~ [batch_size, height * width // patch_size ^ 2, hidden_dim]
        x = x.mean(dim=1) # x ~ [batch_size, hidden_dim]
        logits = self.head(x) # logits ~ [batch_size, num_classes]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return (loss, logits)
