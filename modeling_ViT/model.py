import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        pass
    pass

class ViT(nn.Module):
    def __init__(self, image_size, patch_size)
    super().__init__()
    num_patches = (image_size // patch_size) ** 2
    patch_dim = channels * patch_size ** 2

    self.patch_size = patch_size
    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
    self.patch_to_embedding = nn.Linear(patch_dim, dim)
    self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    self.dropout = nn.Dropout(emb_dropout)


    self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
    self.to_cls_token = nn.Identity()


    self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
            )

    def forward(self, img, mask=None):
        pass

