import torch
import torch.nn as nn


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_embed,
        d_proj,
        div_val=1,
        cutoffs=[512, 1024, 2048],
    ):
        super().__init__()
        self.div_val = div_val
        self.d_embed = d_embed
        self.d_proj = d_proj
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
                self.proj_layers.append(
                    nn.Parameter(torch.FloatTensor(d_proj, d_embed_i))
                )

    def forward(
        self,
        inp,  # inp ~ [batch_size, max_seq_len]
    ):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros(
                [inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device
            )
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = nn.functional.linear(d_proj, self.proj_layers[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed_shape = inp.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)

        embed.mul_(self.emb_scale)
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
