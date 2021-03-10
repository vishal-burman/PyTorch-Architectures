## GPT2

Implementation of GPT2 architecture proposed in the paper _Language Models are Unsupervised Multitask Learners_ by A. Radford et al.

## Usage

```python
import torch
from model import GPT2Classify

x_sample = torch.arange(4).unsqueeze(0).to(torch.long)
att_sample = torch.ones(1, 4, dtype=torch.long)
labels = torch.tensor([1])

model = GPT2Classify()

preds = model(input_ids=x_sample, attention_mask=att_sample, labels=labels)

loss, logits = preds[0], preds[1]
```

## Training

Training procedure is explained in test_sample_GPT2.ipynb

## Citations

```bibtex
@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}
```
