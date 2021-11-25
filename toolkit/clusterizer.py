import logging
import time
from datetime import timedelta
from typing import List, Union

import fire
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
]


def clusterer(
    corpus_sentences: Union[str, List[str]],
    batch_size: int,
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
    convert_to_numpy: bool = True,
    convert_to_tensor: bool = False,
    threshold: float = 0.75,
    min_community_size: int = 10,
    init_max_size: int = 1000,
):
    tokenizer, model = _init_pipeline(model_name)
    model.to(_get_device())

    if isinstance(corpus_sentences, str):
        corpus_sentences = _file_to_corpus(corpus_sentences)

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
        sentences_batch = sentences_sorted[start_index : start_index + batch_size]

        features = tokenizer(
            sentences_batch, padding=True, truncation=True, return_tensors="pt"
        )
        features = _dict_to_device(features, device=_get_device())

        with torch.set_grad_enabled(False):
            outputs = model(**features)
            token_embeddings = outputs[0]
            pooled_embeddings = _mean_pooling(
                token_embeddings, features["attention_mask"]
            )
            pooled_embeddings = pooled_embeddings.detach()
            pooled_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)

        if convert_to_numpy:
            pooled_embeddings = pooled_embeddings.cpu()

        all_embeddings.extend(pooled_embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

    if convert_to_tensor:
        all_embeddings = torch.stack(all_embeddings)
    elif convert_to_numpy:
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

    output = _community_detection(
        all_embeddings,
        threshold=threshold,
        min_community_size=min_community_size,
        init_max_size=init_max_size,
    )
    logger.info(f"Total Clusters: {len(output)}")
    if len(output) > 0:
        logger.info(f"Length of Largest Cluster: {len(output[0])}")
    return output


def _file_to_corpus(filename: str):
    with open(filename, "r") as f:
        corpus = f.readlines()

    corpus = [line.strip() for line in corpus]
    return corpus


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
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


def _dict_to_device(sample_dict, device=torch.device("cpu")):
    keys, values = list(sample_dict.keys()), list(sample_dict.values())
    if not all(isinstance(x, torch.Tensor) for x in values):
        raise TypeError("Only torch.Tensor values can be shifted to CUDA")
    values = list(map(lambda x: x.to(device), values))
    final_dict = dict(zip(keys, values))
    return final_dict


def _community_detection(
    embeddings,
    threshold: float = 0.75,
    min_community_size: int = 10,
    init_max_size: int = 1000,
):
    start_time = time.time()
    init_max_size = min(init_max_size, len(embeddings))  # Max size of community
    logger.info(f"Maximum size of community = {init_max_size}")

    cosine_scores = _calculate_cs(embeddings, embeddings)
    top_k_values, _ = cosine_scores.topk(k=min_community_size, largest=True)

    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            top_val_large, top_idx_large = cosine_scores[i].topk(
                k=init_max_size, largest=True
            )
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:  # start fine-grained search
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break
                    new_cluster.append(idx)
            else:
                # Slow process!
                for idx, val in enumerate(cosine_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(
        extracted_communities, key=lambda x: len(x), reverse=True
    )

    # Step 2 --> Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    end_time = time.time() - start_time
    elapsed_time_str = str(timedelta(milliseconds=end_time))
    logger.info(f"Elapsed Time for clustering: {elapsed_time_str}")
    return unique_communities


def _calculate_cs(a: torch.Tensor, b: torch.Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.Tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.Tensor(b)

    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


if __name__ == "__main__":
    fire.Fire(clusterer)
