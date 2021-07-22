## ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

Implementation of ShuffleNet architecture proposed in the paper _ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices_ by X. Zhang et al.

## Usage

```python
import torch
import torch.nn as nn
from config import ShuffleNetConfig
from model import ShuffleNet

config = ShuffleNetConfig()
model = ShuffleNet(config)

img_sample = torch.rand(2, 3, 224, 224)
labels = torch.tensor([1, 1], dtype=torch.long)

outputs = model(pixel_values=img_sample, labels=labels)
loss, logits = outputs[0], outputs[1]
```

## Training

Training procedure is explained in test_sample_ShuffleNet.ipynb

## Citations

```bibtex
@inproceedings{zhang2018shufflenet,
  title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
  author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6848--6856},
  year={2018}
}
```
