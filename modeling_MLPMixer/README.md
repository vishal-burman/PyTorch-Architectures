## MLP-Mixer: An all-MLP Architecture for Vision

Implementation of the MLPMixer architecture proposed in the paper _MLP-Mixer: An all-MLP Architecture for Vision_ by Ilya Tolstikhin et al.

## Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MLPMixer

model = MLPMixer(image_size=256,
        patch_size=16,
        channel=3,
        dim=512,
        depth=8,
        num_classes=2)

x_sample = torch.rand(2, 3, 256, 256, dtype=torch.float)
logits = model(x_sample) # logits ~ [batch_size, num_classes]
```

## Training

Training procedure is explained in test_sample_MLPMixer.ipynb

## Citations

```bibtex
@article{tolstikhin2021mlp,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and others},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}
```
