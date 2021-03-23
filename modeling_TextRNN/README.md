## Finding Structure in Time

Implementation of TextRNN architecture proposed in the paper _Finding Structure in Time_ by Jeffrey L. Elman.

## Usage

```python
import torch
from model import TextRNN

x_sample = torch.arange(4, dtype=torch.long).reshape(1, 4)
x_sample = torch.eye(100)[x_sample]

model = TextRNN(vocab_size=100, hidden_size=100)

logits = model(x_sample)
```

## Training

Training procedure is explained in test_sample_TextRNN.ipynb

## Citations

```bibtex
@article{elman1990finding,
  title={Finding structure in time},
  author={Elman, Jeffrey L},
  journal={Cognitive science},
  volume={14},
  number={2},
  pages={179--211},
  year={1990},
  publisher={Wiley Online Library}
}
```
