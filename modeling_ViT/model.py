import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.to_out = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Dropout(dropout),
                )

    def forward(self, x, mask=None): # x ~ [batch_size, (img_size // patch_size) + 1, dim] || mask ~ [batch_size, img_size // patch_size, img_size // patch_size]
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0., emb_dropout=0.)
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

    self.proj = nn.Conv2d(in_channels=3, out_channels=patch_dim, kernel_size=self.patch_size, stride=self.patch_size)

    self.layers = nn.ModuleList([])
    for _ in range(depth):
        self.layers.append([
            Residual(Prenorm(dim, Attention(dim, heads=heads, dropout=dropout))),
            Residual(Prenorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
            ])

    def forward(self, img, mask=None): # img ~ [batch_size, channels, height, width] || mask ~ [batch_size, img_size // patch_size, img_size // patch_size]
        x = self.proj(x).flatten(2).transpose(1, 2) # x ~ [batch_size, img_size // patch_size, patch_dim]
        x = self.patch_to_embedding(x) # x ~ [batch_size, img_size // patch_size, dim]
        b, n, _ = x.shape # b ~ batch_size || n ~ img_size // patch_size

        cls_tokens = self.cls_token.expand(b, -1, -1) # cls_tokens ~ [batch_size, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1) # x ~ [batch_size, (img_size // patch_size) + 1, dim]
        x += self.pos_embedding[:, :(n+1)] # x ~ [batch_size, (img_size // patch_size) + 1, dim]
        x = self.dropout(x) # x ~ [batch_size, (img_size // patch_size) + 1, dim]

        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
