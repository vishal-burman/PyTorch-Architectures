## Very Deep Convolutional Networks for Large-Scale Image Recognition

Implementation of VGG16 architecture proposed in the paper _Very Deep Convolutional Networks for Large-Scale Image Recognition_ by K. Simonyan et al.

## Usage

```python
import torch
from model import VGG16

x_sample = torch.rand(1, 3, 32, 32)

model = VGG16(num_classes=2, num_features=x_sample.size(2)*x_sample.size(3))

logits, probs = model(x_sample)
```

## Training

Training procedure is explained in VGG16-CIFAR10.ipynb

## Citations

```bibtex
@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}
```
