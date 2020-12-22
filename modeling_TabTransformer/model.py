import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class Transformer(nn.Module):
    def __init__(self,):
        pass
    def forward(self):
        pass

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
    
    def forward(self, x_categ, x_cont):
        assert x_categ.shape[-1] == self.num_categories
        x_categ += self.categories_offset
        x = self.transformer(x_categ)
        flat_categ = x.flatten(1)
        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std
        normed_cont = self.norm(x_cont)
        x = torch.cat((flat_categ,normed_cont), dim=-1)
        return self.mlp(x)
