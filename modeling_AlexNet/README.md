## AlexNet

Implementation of AlexNet architecture proposed by Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton.

## Usage

```python
import torch
from model import AlexNet

x_sample = torch.rand(2, 3, 128, 128)
labels_sample = torch.ones(2, dtype=torch.long)

model = AlexNet(num_classes = 2)

loss, logits = model(pixel_values=x_sample, labels=labels_sample)
```

## Training

Training procedure is exaplained in test_sample_AlexNet.ipynb

## Citations

```bibtext
@inproceedings{NIPS2012_c399862d,
 author = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {F. Pereira and C. J. C. Burges and L. Bottou and K. Q. Weinberger},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {ImageNet Classification with Deep Convolutional Neural Networks},
 url = {https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf},
 volume = {25},
 year = {2012}
}
```
