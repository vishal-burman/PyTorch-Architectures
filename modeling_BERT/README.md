## BERT: Bidirectional Encoder Representations from Transformers

Implementation of BERT architecture proposed by Jacob Devlin et al.

## Usage

```python
import torch
from config_bert import BertConfig
from model import BertClassify

config = BertConfig()

x_sample = torch.ones(1, 8, dtype=torch.long)
att_sample = torch.ones(1, 8, dtype=torch.long)
y_sample = torch.tensor([1])

model = BertClassify(config)

preds = model(input_ids=x_sample, attention_mask=att_sample, labels=y_sample)
```

## Training

Training procedure is explained in test_sample_BERT.ipynb

## Citations

```bibtex
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```
