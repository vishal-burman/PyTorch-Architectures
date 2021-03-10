## NiN: Network in Network

Implementation of NiN architecture proposed in the paper _Network in Network_ by M. Lin et al.

## Usage

```python
import torch
from model import NiN

x_sample = torch.rand(1, 3, 32, 32)

model = NiN(num_classes=2)

preds = model(x_sample)
```

## Training

Training procedure is explained in test_sample_NiN.ipynb

## Citations

```bibtex
@article{lin2013network,
  title={Network in network},
  author={Lin, Min and Chen, Qiang and Yan, Shuicheng},
  journal={arXiv preprint arXiv:1312.4400},
  year={2013}
}
```
