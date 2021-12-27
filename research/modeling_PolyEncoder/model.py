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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_pooled: bool = False,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state
        if return_pooled:
            logits = logits[:, 0]  # First token pooling
        return logits


class PolyEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        poly_m: int,
    ):
        super().__init__()
        self.encoder = Encoder(encoder_name=encoder_name)
        self.poly_m = poly_m
        pass

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ):
        context_emb = self.encoder(input_ids=context_ids, attention_mask=context_mask)
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(
            context_emb.device
        )
        pass
