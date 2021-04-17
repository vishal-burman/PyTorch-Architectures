## Sequence to Sequence Learning with Neural Networks

Implementation of Seq2Seq architecture proposed in the paper _Sequence Learning with Neural Networks_ by I. Sutskevar et al.

## Usage

```python
import torch
from model import Encoder, Decoder, Seq2Seq

enc = Encoder(input_dim=100,
        emb_dim=32,
        hidden_dim=200,
        num_layers=4,
        p_drop=0.1,
        )

dec = Decoder(output_dim=100,
        emb_dim=32,
        hidden_dim=200,
        num_layers=4,
        p_drop=0.1,
        )

model = Seq2Seq(encoder=enc, decoder=dec)

x_sample = torch.arange(4, dtype=torch.long).reshape(4, 1)

output = model(src=x_sample, trg=x_sample)
```

## Training

Training procedure is explained in test_sample_Seq2Seq.ipynb

## Citations

```bibtex
@article{sutskever2014sequence,
  title={Sequence to sequence learning with neural networks},
  author={Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V},
  journal={arXiv preprint arXiv:1409.3215},
  year={2014}
}
```
