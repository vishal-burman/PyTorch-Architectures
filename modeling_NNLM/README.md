## NNLM: Neural Probabilistic Language Model

Implementation of NNLM architecture proposed in the paper _A Neural Probablistic Language Model_ by Y. Bengio et al.

## Usage

```python
import torch
from model import NNLM

x_sample = torch.arange(2).reshape(1, 2).to(torch.long)

model = NNLM(n_class=4, m=50, n_hidden=256, n_step=2)

preds = model(x_sample)
```

## Training

Training procedure is explained in test_sample_NNLM.ipynb

## Citations

```bibtex
@article{bengio2003neural,
  title={A neural probabilistic language model},
  author={Bengio, Yoshua and Ducharme, R{\'e}jean and Vincent, Pascal and Janvin, Christian},
  journal={The journal of machine learning research},
  volume={3},
  pages={1137--1155},
  year={2003},
  publisher={JMLR. org}
}
```
