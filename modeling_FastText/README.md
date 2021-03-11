## FastText

Implementation of FastText architecture proposed in the paper _Bag of Tricks for Efficient Text Classification_ by A. Joulin et al.

## Usage

```python
import torch
from model import FastText

x_sample = torch.arange(3).unsqueeze(0).to(torch.long)

model = FastText(vocab_size=10,
		 embedding_size=8,
		 hidden_size=100,
		 output_size=2,
		 padding_idx=0)

preds = model(x_sample)
```

## Training

Training procedure is explained in test_sample_FastText.ipynb

## Citations

```bibtex
@article{joulin2016bag,
  title={Bag of tricks for efficient text classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```
