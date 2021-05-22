class FNetConfig:
    def __init__(self,
            vocab_size=30522,
            embed_dim=768,
            max_position_embed=512,
            padding_idx=0,
            p_drop=0.1):

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_position_embed = max_position_embed
        self.padding_idx = padding_idx
        self.p_drop = p_drop
