import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Encode(nn.Module):
    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if "max_position_embeddings" in self.encoder.config.to_dict():
            self.max_length = self.encoder.config.max_position_embeddings
        else:
            raise NotImplementedError

    def forward(
        self,
        text: str,
    ):
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.encoder(**tokens)
        logits = outputs.last_hidden_state
        pooled_output = logits[:, 0]
        return pooled_output
