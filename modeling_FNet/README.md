## FNet: Mixing Tokens with Fourier Transforms

Implementation of FNet architecture proposed in the paper _FNet: Mixing Tokens with Fourier Transforms_ by J. Lee-Thorp et al.

## Usage

```python
import torch
from config import FNetConfig
from model import FNetClassify

config = FNetConfig()
model = FNetClassify(config)

input_ids = torch.arange(16, dtype=torch.long).reshape(2, 8)
labels = torch.tensor([0, 1])

outputs = model(input_ids=input_ids, labels=labels)
loss, logits = outputs[0], outputs[1]
```

## Training

Training procedure is explained in test_sample_FNet.ipynb

## Citations

```bibtex
@article{lee2021fnet,
  title={FNet: Mixing Tokens with Fourier Transforms},
  author={Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and Ontanon, Santiago},
  journal={arXiv preprint arXiv:2105.03824},
  year={2021}
}
```
