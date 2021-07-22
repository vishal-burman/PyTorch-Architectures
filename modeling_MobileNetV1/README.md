## MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

Implementation of MobileNetV1 architecture proposed in the paper _MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications_ by Andrew G. et al.

## Usage

```python
import torch
from config import MobileNetV1Config
from model import MobileNetV1

config = MobileNetV1Config()
model = MobileNetV1(config)

img_sample = torch.rand(2, 3, 224, 224)
labels = torch.tensor([1, 1], dtype=torch.long)

outputs = model(pixel_values=img_sample, labels=labels)
loss, logits = outputs[0], outputs[1]
```

## Training

Training details are explained in test_sample_MobileNetV1.ipynb

## Citations

```bibtex
@article{DBLP:journals/corr/HowardZCKWWAA17,
  author    = {Andrew G. Howard and
               Menglong Zhu and
               Bo Chen and
               Dmitry Kalenichenko and
               Weijun Wang and
               Tobias Weyand and
               Marco Andreetto and
               Hartwig Adam},
  title     = {MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
               Applications},
  journal   = {CoRR},
  volume    = {abs/1704.04861},
  year      = {2017},
  url       = {http://arxiv.org/abs/1704.04861},
  archivePrefix = {arXiv},
  eprint    = {1704.04861},
  timestamp = {Thu, 27 May 2021 16:20:51 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HowardZCKWWAA17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
