class TransformerXLConfig:
    def __init__(
        self,
        vocab_size=267735,
        cutoffs=[20000, 40000, 200000],
        d_model=1024,
        d_embed=1024,
        div_val=4,
        sample_softmax=-1,
    ):
        self.vocab_size = vocab_size
        self.cutoffs = cutoffs
        self.d_model = d_model
        self.d_embed = d_embed
        self.div_val = div_val
        self.sample_softmax = sample_softmax
