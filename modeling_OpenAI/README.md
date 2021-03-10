## Improving Language Understanding By Generative Pre-Training

Implementation of the original GPT architecture proposed in the paper _Improving Language Understanding By Generative Pre-Training_ by A. Radford et al.

## Usage

```python
import torch
from model import OpenAIGPTLMHeadModel
from config_openai import OpenAIGPTConfig

config = OpenAIGPTConfig()

model = OpenAIGPTLMHeadModel(config)

x_sample = torch.arange(4).reshape(1, 4).to(torch.long)
att_sample = torch.ones(1, 4, dtype=torch.long)
labels = torch.arange(4).reshape(1, 4).to(torch.long)

preds = model(input_ids=x_sample, attention_mask=att_sample, labels=labels)

loss, logits = preds[0], preds[1]
```

## Training

Training procedure is explained in test_sample_OpenAI.ipynb

## Citations

```bibtex
@article{radford2018improving,
  title={Improving language understanding by generative pre-training},
  author={Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya},
  year={2018}
}
```
