class XLNetConfig:
    def __init__(
            self,
            vocab_size=32000,
            d_model=1024,
            n_layer=4,
            n_head=16,
            d_inner=1024,
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            dropout=0.1,
            num_labels=2,
            ):
        self.vocab_size = vocab_size
        self.d_head = d_model // n_head
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_inner = d_inner
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.dropout = dropout
        self.num_labels = num_labels
