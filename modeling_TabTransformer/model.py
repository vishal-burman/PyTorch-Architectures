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
