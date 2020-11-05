import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
                )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim*3, bias=False)
        self.to_out = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Dropout(dropout),
                )

    def forward(self, x): # x ~ [batch_size, (img_size // patch_size) + 1, dim] || mask ~ [batch_size, img_size // patch_size, img_size // patch_size]
        b, n, dim, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1) # tuple(3 items) item ~ [batch_size, (img_size) // patch_size) + 1, dim]
        q, k, v = map(lambda t: t.reshape(b, n, h, dim // h).transpose(1, 2), qkv)
        dots = (q @ k.transpose(-2, -1)) * self.scale

        attn = dots.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)
        return out

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0., emb_dropout=0.):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes),
                )

        self.proj = nn.Conv2d(in_channels=3, out_channels=patch_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                ]))

    def forward(self, img): # img ~ [batch_size, channels, height, width]
        x = self.proj(img).flatten(2).transpose(1, 2) # x ~ [batch_size, img_size // patch_size ^ 2, patch_dim]
        x = self.patch_to_embedding(x) # x ~ [batch_size, img_size // patch_size, dim]
        b, n, _ = x.shape # b ~ batch_size || n ~ img_size // patch_size

        cls_tokens = self.cls_token.expand(b, -1, -1) # cls_tokens ~ [batch_size, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1) # x ~ [batch_size, (img_size // patch_size) + 1, dim]
        x += self.pos_embedding[:, :(n+1)] # x ~ [batch_size, (img_size // patch_size) + 1, dim]
        x = self.dropout(x) # x ~ [batch_size, (img_size // patch_size) + 1, dim]

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
