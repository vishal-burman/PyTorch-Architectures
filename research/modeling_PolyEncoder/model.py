from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class Encode(nn.Module):
    def __init__(
        self,
        encoder_name: str,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)

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
        batch_size: int,
        poly_m: int,
        hidden_size: int,
        encoder_name: str = "distilbert-base-uncased",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = Encode(encoder_name=encoder_name)
        self.poly_m = poly_m
        self.poly_code_embeddings = nn.Embedding(self.poly_m, hidden_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, hidden_size ** -0.5)
        self.pre_classifier = nn.Linear(batch_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def dot_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        attention_weights = query @ key.transpose(1, -1)
        attention_weights = F.softmax(attention_weights, dim=-1)
        output = attention_weights @ value
        return output

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        bs, seq_len = context_ids.shape

        # Context Encoder
        context_emb = self.encoder(input_ids=context_ids, attention_mask=context_mask)
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(
            context_emb.device
        )
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(bs, self.poly_m)
        poly_code_emb = self.poly_code_embeddings(poly_code_ids)
        embs = self.dot_attention(
            query=poly_code_emb, key=context_emb, value=context_emb
        )  # [bs, poly_m, dim]

        # Candidate Encoder
        candidate_emb = self.encoder(
            input_ids=candidate_ids, attention_mask=candidate_mask, return_pooled=True
        )
        candidate_emb = candidate_emb.unsqueeze(1)
        candidate_emb = candidate_emb.permute(1, 0, 2).expand(
            bs, bs, candidate_emb.size(2)
        )

        weighted_embs = self.dot_attention(query=candidate_emb, key=embs, value=embs)
        dot_product = (weighted_embs * candidate_emb).sum(-1)  # [bs, bs]

        logits = self.pre_classifier(dot_product)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.classifier(logits)

        if labels is not None:
            # TODO create classification head for training
            pass
        else:
            # TODO return logits for inference
            pass

        return logits
