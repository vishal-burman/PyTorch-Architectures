import torch
import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        # w ~ [nx, nf]
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        # self.weight ~ [nx, nf]
        self.weight = nn.Parameter(w)
        # self.bias ~ [1, nf]
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        # size_out ~ [batch_size, max_len, nf]
        size_out = x.size()[:-1] + (self.nf,)
        # x ~ [batch_size * max_len, nf]
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        # x ~ [batch_size, max_len, nf]
        x = x.view(*size_out)
        return x

class PretrainedModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config
    
    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        Prepare the head_mask if needed
        """
        head_mask = [None] * num_hidden_layers
        return head_mask

    def init_weights(self):
        # Initialize the weights
        self.apply(self._init_weights)

        # Tie weights if necessary
        self.tie_weights()

    def tie_weights(self):
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight
        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                    output_embeddings.bias.data,
                    (
                        0,
                        output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                        ),
                    "constant",
                    0,
                    )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def get_output_embeddings(self):
        return None

    def get_input_embeddings(self):
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
