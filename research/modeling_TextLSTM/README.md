## Long Short-Term Memory

Implementation of a simple architecture for causal language modeling using LSTM

## Usage

```python
import torch
from model import TextLSTM

model = TextLSTM(
        vocab_size=100,
        embedding_size=32,
        hidden_size=100,
        padding_idx=0,
        )

x_sample = torch.ones(1, 3, dtype=torch.long)

logits = model(x_sample)
```

## Training

Training procedure is explained in test_sample_TextLSTM.ipynb

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
