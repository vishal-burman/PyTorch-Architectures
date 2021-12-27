import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Encode(nn.Module):
    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        return_pooled: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if "max_position_embeddings" in self.encoder.config.to_dict():
            self.max_length = self.encoder.config.max_position_embeddings
        else:
            raise NotImplementedError
        self.return_pooled = return_pooled

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
        if self.return_pooled:
            logits = logits[:, 0]  # First token pooling
        return logits
