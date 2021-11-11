## LeNet

Implementation of LeNet architecture proposed in the paper _LeNet-5_ by Yann LeCun et al.

## Unit Tests

```bash
python -m unittest test_model.py
```

## Usage

```python
import torch
from model import LeNet

x_sample = torch.rand(3, 3, 32, 32)
labels = torch.ones((1, 3), dtype=torch.long)

model = LeNet(num_classes=2, grayscale=False)

loss, logits = model(x_sample, labels)
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
