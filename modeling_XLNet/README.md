## XLNet: Generalized Autoregressive Pretraining for Language Understanding

Implementation of XLNet architecture proposed in the paper _XLNet: Generalized Autoregressive Pretraining for Language Understanding_ by Z. Yang et al.

## Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import XLNetClassify
from config import XLNetConfig

config = XLNetConfig()
config.num_labels = 2

input_ids = torch.arange(8, dtype=torch.long).reshape(2, 4)
attention_mask = torch.ones((2, 4), dtype=torch.long)
labels = torch.zeros((2, 1), dtype=torch.long)
assert input_ids.shape == attention_mask.shape

model = XLNetClassify(config)
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
logits, loss = outputs[0], outputs[1]
```

## Training

Training procedure is explained in test_sample_XLNet.ipynb

## Citations

```bibtex
@article{yang2019xlnet,
  title={Xlnet: Generalized autoregressive pretraining for language understanding},
  author={Yang, Zhilin and Dai, Zihang and Yang, Yiming and Carbonell, Jaime and Salakhutdinov, Ruslan and Le, Quoc V},
  journal={arXiv preprint arXiv:1906.08237},
  year={2019}
}
```
