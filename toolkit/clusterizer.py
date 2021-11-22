import logging
from typing import List, Union

import fire
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

from .utils import dict_to_device

logger = logging.get_logger(__name__)

SUPPORTED_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
]


def clusterer(
    corpus_sentences: List[str],
    batch_size: int,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
):
    tokenizer, model = _init_pipeline(model_name)
    model.to(_get_device())

    if len(corpus_sentences) <= 1:
        raise ValueError(
            f"Clusterizer cannot perform with {len(corpus_sentences)} sentences"
        )

    length_sorted_idx = np.argsort(
        [-_get_length(sentence) for sentence in corpus_sentences]
    )
    sentences_sorted = [corpus_sentences[idx] for idx in length_sorted_idx]

    all_embeddings = []
    for start_index in trange(0, len(corpus_sentences), batch_size, desc="Batches"):
        sentences_batch = corpus_sentences[start_index : start_index + batch_size]

        features = tokenizer(
            sentences_batch, padding=True, truncation=True, return_tensors="pt"
        )
        features = dict_to_device(features, device=_get_device())

        with torch.inference_mode():
            outputs = model(**features)
            token_embeddings = outputs[0]
            pooled_embeddings = mean_pooling(
                token_embeddings, features["attention_mask"]
            )
            pooled_embeddings = pooled_embeddings.detach()
            pooled_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
            pooled_embeddings = pooled_embeddings.detach()

        if convert_to_numpy:
            pooled_embeddings = pooled_embeddings.cpu()

        all_embeddings.extend(pooled_embeddings)


def _get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    device = torch.device("cpu")
    return device


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def _init_pipeline(model_name: str):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Please try clusterizer.py with {SUPPORTED_MODELS}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Parameters = {params}")

    return tokenizer, model


def _get_length(text: Union[str, List[str]]):
    return sum([len(t) for t in text])


if __name__ == "__main__":
    fire.Fire(clusterer)