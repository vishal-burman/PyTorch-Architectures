## Cross-lingual Language Model Pretraining

Implementation of XLM architecture proposed in the paper _Cross-lingual Language Model Pretraining_ by G. Lample et al.

## Usage

```python
import torch
from config_xlm import XLMConfig
from model import XLMWithLMHeadModel

config = XLMConfig()
model = XLMWithLMHeadModel(config)

x_sample = torch.arange(3, dtype=torch.long).reshape(1, 3)
att_sample = torch.ones(1, 3, dtype=torch.long)

preds = model(input_ids=x_sample, attention_mask=att_sample, labels=x_sample)
loss, logits = preds[0], preds[1]
```

## Training

Training procedure is explained in test_sample_XLM.ipynb

## Citations

```bibtex
@article{lample2019cross,
  title={Cross-lingual language model pretraining},
  author={Lample, Guillaume and Conneau, Alexis},
  journal={arXiv preprint arXiv:1901.07291},
  year={2019}
}
```
