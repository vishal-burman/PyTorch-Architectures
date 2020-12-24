import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_heads ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # x ~ [batch_size, num_categ, dim]
        h = self.heads
        bs, num_categ = x.shape[:2] # bs ~ batch_size, num_categ ~ num_categ
        q, k, v = self.to_qkv(x).chunk(3, dim=-1) # q, k, v ~ [batch_size, num_categ, inner_dim]
        q, k, v = map(lambda x: x.reshape(bs, num_categ, h, -1).transpose(1, 2), (q, k, v)) # q, k, v ~ [batch_size, heads, num_categ, inner_dim//heads]
        sim = (q @ k.transpose(-1, -2)) * self.scale # sim ~ [batch_size, heads, num_categ, num_categ]
        attn = sim.softmax(dim=-1) # attn ~ [batch_size, heads, num_categ, num_categ]
        attn = self.dropout(attn) # attn ~ [batch_size, heads, num_categ, num_categ]
        out =  attn @ v # out ~ [batch_size, heads, num_categ, inner_dim//heads]
        out = out.reshape(bs, num_categ, -1) # [batch_size, num_categ, inner_dim]
        out = self.to_out(out) # out ~ [batch_size, num_categ, dim]
        return out

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Prenorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                Residual(Prenorm(dim, FeedForward(dim, dropout = ff_dropout))),
                ]))

    def forward(self, x): # x ~ [batch_size, num_categ]
        x = self.embeds(x) # x ~ [batch_size, num_categ, dim]

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x

# TODO enclose init in config
class TabTransformer(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head = 16,
            dim_out = 1,
            mlp_hidden_mults = (4, 2),
            mlp_act = None,
            num_special_tokens = 2,
            continuous_mean_std = None,
            attn_dropout = 0.,
            ff_dropout = 0.,
            ):
        super().__init__()
        assert(map(lambda x : x > 0, categories)), 'Number of categories must be +ve'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)
        if exists(continuous_mean_std):
            self.register_buffer('continuous_mean_std', continuous_mean_std)
        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.transformer = Transformer(
                num_tokens = num_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                )
        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8
        hidden_dimensions = list(map(lambda x: l * x, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act = mlp_act)
    
    def forward(self, x_categ, x_cont): # x_categ ~ [batch_size, num_categ], x_cont ~ [batch_size, num_cont]
        assert x_categ.shape[-1] == self.num_categories
        x_categ += self.categories_offset # x_categ ~ [batch_size, x_categ]
        x = self.transformer(x_categ)
        flat_categ = x.flatten(1)
        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std
        normed_cont = self.norm(x_cont)
        x = torch.cat((flat_categ,normed_cont), dim=-1)
        return self.mlp(x)
