import torch
import torch.nn as nn


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_embed,
        div_val=1,
        cutoffs=[512, 1024, 2048],
    ):
        super().__init__()
        self.div_val = div_val
        self.d_embed = d_embed
        self.cutoffs = cutoffs + [vocab_size]
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(vocab_size, d_embed))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_embed_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_embed_i))

    def forward(
        self,
        inp,  # inp ~ [batch_size, max_seq_len]
    ):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
        else:
            raise NotImplementedError
        return embed


def _test():
    print("Testing for div_val = 1")

    d_embed = 512
    vocab_size = 128
    model = AdaptiveEmbedding(vocab_size=vocab_size, d_embed=d_embed)
    model.eval()
    with torch.set_grad_enabled(False):
        sample_inp = torch.arange(16, dtype=torch.long).reshape(2, 8)
        output = model(sample_inp)
    assert output.dim() == 3, f"Output dim is {output.dim()} instead of 3"
    assert (
        output.size(2) == d_embed
    ), f"Wrong output shape: {output.size(2)} instead of {d_embed}"
    print("Testing for div_val = 1 done!")


if __name__ == "__main__":
    _test()
