## Convolutional Neural Networks for Sentence Classification

Implementation of TextCNN architecture proposed in the paper _Convolutional Neural Networks for Sentence Classification_ by Yoon Kim.

## Usage

```python
import torch
import torch.nn as nn
from model import TextCNN

model = TextCNN(
		num_filters=3, 
		filter_sizes=[2, 2, 2], 
		vocab_size=100, 
		embedding_size=100, 
		sequence_length=3,
		)

x_sample = torch.arange(6, dtype=torch.long).reshape(2, 3)

logits = model(x_sample)
```

## Training

Training procedure is explained in test_sample_TextCNN.ipynb

## Citations

```bibtex
@article{DBLP:journals/corr/Kim14f,
  author    = {Yoon Kim},
  title     = {Convolutional Neural Networks for Sentence Classification},
  journal   = {CoRR},
  volume    = {abs/1408.5882},
  year      = {2014},
  url       = {http://arxiv.org/abs/1408.5882},
  archivePrefix = {arXiv},
  eprint    = {1408.5882},
  timestamp = {Mon, 13 Aug 2018 16:46:21 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/Kim14f.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
