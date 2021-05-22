class FNetConfig:
    def __init__(self,
            vocab_size=30522,
            dim=768,
            expanded_dim=2048,
            max_position_embed=512,
            padding_idx=0,
            p_drop=0.1,
            depth=8,
            eps=1e-12,):

        self.vocab_size = vocab_size
        self.dim = dim
        self.expanded_dim = expanded_dim
        self.max_position_embed = max_position_embed
        self.padding_idx = padding_idx
        self.p_drop = p_drop
        self.depth = depth
        self.eps = eps
