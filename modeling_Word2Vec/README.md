## Distributed Representations of Words and Phrases and their Compositionality

Implementation of Word2Vec architecture proposed in the paper _Distributed Representations of Words and Phrases and their Compositionality_ by T. Mikolov et al.

## Usage

```python
import torch
from model import Word2Vec

x_sample = torch.arange(7).reshape(1, 7).to(torch.float)

preds = model(x_sample)
```

## Training

Training procedure is explained in test_sample_Word2Vec.ipynb

## Citations

```bibtex
@article{mikolov2013distributed,
  title={Distributed representations of words and phrases and their compositionality},
  author={Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  journal={arXiv preprint arXiv:1310.4546},
  year={2013}
}
```
