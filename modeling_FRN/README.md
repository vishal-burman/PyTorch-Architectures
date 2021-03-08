## FRN: Filter Response Normalization

Implementation of FRN architecture for improving performance using small mini-batch sizes over Batch Normalization

## Usage

```python
import torch
import torch.nn as nn

from FRN import FilterResponseNormalization

class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.frn = FilterResponseNormalization(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.frn(x)
        return x

model = CustomModule()

x = torch.rand(2, 3, 32, 32)
preds = model(x)
```

## Training

Training procedure is explained in test_sample_FRN.ipynb

## Citations

```bibtex
@inproceedings{singh2020filter,
  title={Filter response normalization layer: Eliminating batch dependence in the training of deep neural networks},
  author={Singh, Saurabh and Krishnan, Shankar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11237--11246},
  year={2020}
}
```
