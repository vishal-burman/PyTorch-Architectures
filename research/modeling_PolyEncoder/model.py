import torch
import torch.nn as nn
from transformers import AutoModel


class Encode(nn.Module):
    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if "max_position_embeddings" in self.encoder.config.to_dict():
            self.max_length = self.encoder.max_position_embeddings
        else:
            raise NotImplementedError

    def forward(
        self,
        text: str,
    ):
        pass
