## Structured Self-Attentive Sentence Embedding

Implementation of BiLSTMSE architecture proposed in the paper _A Structured Self-Attentive Sentence Embedding_ by Z. Lin et al.

## Usage

```python
import torch
from model import BiLSTMSE

model = BiLSTMSE(
        vocab_size=5,
        emb_dim=300,
        hidden_dim=300,
        n_layers=2,
        natt_unit=300,
        natt_hops=1,
        nfc=512,
        n_class=2,
        drop_prob=0.5)

x_sample = torch.arange(3, dtype=torch.long).reshape(1, 3)

logits, att_weights = model(x_sample)
```

## Training

Training procedure is explained in test_sample_SSE.ipynb

## Citations

```bibtex
@article{lin2017structured,
  title={A structured self-attentive sentence embedding},
  author={Lin, Zhouhan and Feng, Minwei and Santos, Cicero Nogueira dos and Yu, Mo and Xiang, Bing and Zhou, Bowen and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1703.03130},
  year={2017}
}
```
