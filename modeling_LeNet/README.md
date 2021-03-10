## LeNet

Implementation of LeNet architecture proposed in the paper _LeNet-5_ by Yann LeCun et al.

## Usage

```python
import torch
from model import LeNet

x_sample = torch.rand(1, 1, 32, 32)

model = LeNet(num_classes=2, grayscale=True)

logits, pros = model(x_sample)
```

## Training

Training procedure is explained in test_sample_LeNet.ipynb

## Citations

```bibtex
@article{lecun2015lenet,
  title={LeNet-5, convolutional neural networks},
  author={LeCun, Yann and others},
  journal={URL: http://yann. lecun. com/exdb/lenet},
  volume={20},
  number={5},
  pages={14},
  year={2015}
}
```
