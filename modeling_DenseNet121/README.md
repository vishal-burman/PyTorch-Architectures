## DenseNet121

Implementation of DenseNet architecture proposed by Gao Huang et al.

## Usage

```python
import torch
from model import DenseNet121

x_sample = torch.rand(1, 3, 128, 128)

model = DenseNet121(num_classes=2, grayscale=False)

logits, probs = model(x_sample)
```

## Training

Training procedure is explained in test_sample_DenseNet121.ipynb

## Citations

```bibtex
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}
```
