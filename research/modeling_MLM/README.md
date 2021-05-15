## Masked Language Modeling

Implementation of the masked-language modeling proposed in the paper _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_ by Devlin et al.

## Usage

```python
import torch
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from model import MLM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = DistilBertConfig()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
transformer = DistilBertModel(config)

model = MLM(transformer=transformer,
            pad_token_id=tokenizer.pad_token_id,
	    mask_token_id=tokenizer.mask_token_id,
	    mask_prob=0.15,
	    num_tokens=tokenizer.vocab_size,
	    replace_prob=0.9)
model.to(device)

x_sample = torch.arange(8).reshape(1, 8).to(device)

logits, labels = model(x_sample)
```

## Training

Training procedure is explained in test_sample_MLM.ipynb

## Citations

```bibtex
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```
