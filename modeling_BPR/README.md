## BPR: Bayesian Personalized Ranking from Implicit Feedback

Implementation of BPR architecture proposed by Steffen Rendle et al.

## Usage

```python
import torch
from model import BPR
samples = [
        {
            'user_id': torch.tensor([1, 2]),
            'pos_item_id': torch.tensor([1, 2]),
            'neg_item_id': torch.tensor([3, 4]),
            },
        {
            'user_id': torch.tensor([2, 3]),
            'pos_item_id': torch.tensor([3, 4]),
            'neg_item_id': torch.tensor([1, 2]),
            }
        ]

model = BPR(n_users=5, n_items=10, embedding_size=64)
scores = model.predict(samples[1])
```

## Training

Training procedure is explained in test_sample_BPR.ipynb

## Citations

```bibtex
@article{rendle2012bpr,
  title={BPR: Bayesian personalized ranking from implicit feedback},
  author={Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:1205.2618},
  year={2012}
}
```
