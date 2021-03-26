## Causal Language-Model using BiLSTM

Implementation of a simple architecture using BiLSTM for causal language-modeling.

## Usage

```python
import torch
from model import TextBiLSTM

model = TextBiLSTM(
        vocab_size=100,
        embedding_size=32,
        hidden_size=100,
        padding_idx=0,
        )

x_sample = torch.arange(3, dtype=torch.long).reshape(1, 3)

logits = model(x_sample)
```

## Training

Training procedure is explained in test_sample_TextBiLSTM.ipynb

## Citations

```bibtex
@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997},
  publisher={MIT Press}
}
```
