## Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

Implementation of Seq2SeqPR architecture proposed in the paper _Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation_ by Cho et al.

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

Training procedure is explained in test_sample_Seq2SeqPR.ipynb

## Citations

```bibtex
@article{cho2014learning,
  title={Learning phrase representations using RNN encoder-decoder for statistical machine translation},
  author={Cho, Kyunghyun and Van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1406.1078},
  year={2014}
}
```
