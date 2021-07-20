## MobileNetV2: Inverted Residuals and Linear Bottlenecks

Implementation of MobileNetV2 architecture proposed in the paper _MobileNetV2: Inverted Residuals and Linear Bottlenecks_ by Mark Sandler et al.

## Usage

```python
import torch
from model import MobileNetV2
from config import MobileNetV2Config
config = MobileNetV2Config()

img_sample = torch.rand(2, 3, 224, 224)
labels = torch.ones((2, 1), dtype=torch.long)

model = MobileNetV2(config)
outputs = model(pixel_values=img_sample, labels=labels)
loss, logits = outputs[0], outputs[1]
```

## Training

Training procedure explained in test_sample_MobileNetV2.ipynb

## Citations

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```
