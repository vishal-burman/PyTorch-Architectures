import torch

from model import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
x = torch.rand(1, 3, 256, 256)
x = v(x)
print("Done")