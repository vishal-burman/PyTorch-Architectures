import torch
import torch.nn as nn
import torch.nn.functional as F

def default(val, d):
    return val if val is not None else d

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

class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1) # x, gates ~ [batch_size, num_categ, dim * mult]
        return x * F.gelu(gates) # return ~ [batch_size, num_categ, dim * mult]

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim, dim * mult * 2),
                GEGLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * mult, dim),
                )

    def forward(self, x): # x ~ [batch_size, num_categ, dim]
        return self.net(x) # return ~ [batch_size, num_categ, dim]

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
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
        out =  (attn @ v).transpose(1, 2) # out ~ [batch_size, heads, num_categ, inner_dim//heads]
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
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
                ]))

    def forward(self, x): # x ~ [batch_size, num_categ]
        x = self.embeds(x) # x ~ [batch_size, num_categ, dim]
        for attn, ff in self.layers:
            x = attn(x) # x ~ [batch_size, num_categ, dim]
            x = ff(x) # x ~ [batch_size, num_categ, dim]
        return x # return ~ [batch_size, num_categ, dim]

class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)
            if is_last:
                continue
            act = default(act, nn.ReLU())
            layers.append(act)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x): # x ~ [batch_size, num_categ * dim + num_cont]
        return self.mlp(x) # x ~ [batch_size, dim_out]

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
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens) # categories_offset ~ [num_categ + 1]
        categories_offset = categories_offset.cumsum(dim=-1)[:-1] # categories_offset ~ [num_categ]
        self.register_buffer('categories_offset', categories_offset)
        if continuous_mean_std is not None:
            self.register_buffer('continuous_mean_std', continuous_mean_std)
        else:
            self.register_buffer('continuous_mean_std', None)
        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.transformer = Transformer(
                num_tokens = total_tokens,
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
        x_categ += self.categories_offset # x_categ ~ [batch_size, num_categ]
        x = self.transformer(x_categ) # x ~ [batch_size, num_categ, dim]
        flat_categ = x.flatten(1) # flat_categ ~ [batch_size, num_categ * dim]
        if self.continuous_mean_std is not None:
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std # x_cont ~ [batch_size, num_cont]
        normed_cont = self.norm(x_cont) # normed_cont ~ [batch_size, num_cont]
        x = torch.cat((flat_categ, normed_cont), dim=-1) # x ~ [batch_size, (num_categ * dim) + num_cont]
        return self.mlp(x) # return ~ [batch_size, dim_out]
