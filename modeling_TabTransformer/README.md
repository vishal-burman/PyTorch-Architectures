## TabTransformer

Implementation of TabTransformer architecture proposed in the paper _TabTransformer: Tabular Data Modeling using Contextual Embeddings_ by X. Huang et al.

## Usage

```python
import torch
from model import TabTransformer

cont_mean_std = torch.randn(10, 2)

model = TabTransformer(
    categories = (10, 5, 6, 5, 8),      # tuple containing the number of unique values within each category
    num_continuous = 10,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = torch.nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
)

x_categ = torch.randint(0, 5, (1, 5))     # category values, from 0 - max number of categories, in the order as passed into the constructor above
x_cont = torch.randn(1, 10)               # assume continuous values are already normalized individually

pred = model(x_categ, x_cont)
print(pred.shape)
```

## Training

Training procedure is explained in test_sample_TabTransformer.ipynb

## Citations

```bibtex
@article{huang2020tabtransformer,
  title={TabTransformer: Tabular Data Modeling Using Contextual Embeddings},
  author={Huang, Xin and Khetan, Ashish and Cvitkovic, Milan and Karnin, Zohar},
  journal={arXiv preprint arXiv:2012.06678},
  year={2020}
}
```
