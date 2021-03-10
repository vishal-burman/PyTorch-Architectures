## ResNet: Deep Residual Learning for Image Recognition

Implementation of ResNet architecture proposed in the paper _Deep Residual Learning for Image Recognition_ by K. He et al.

## Usage

```python
import torch
from model import ResNet, BasicBlock

model = ResNet(block=BasicBlock,
	       layers=[1, 1, 1, 1],
               num_classes=2,
               grayscale=True,
	 )

x_sample = torch.rand(2, 1, 28, 28)

logits, probs = model(x_sample)
```

```python
import torch
from model_large import ResNet, Bottleneck

model = ResNet(block=Bottleneck,
	       layers=[1, 1, 1, 1],
	       num_classes=2,
	       grayscale=True)

x_sample = torch.rand(2, 1, 128, 128)

logits, probs = model(x_sample)
```

## Training

Training procedure is explained in test_sample_ResNet18.ipynb

## Citations

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```
