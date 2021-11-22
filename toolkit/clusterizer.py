import logging

import fire
from tqdm import trange

logger = logging.get_logger(__name__)


def clusterer(corpus_sentences: List[str], batch_size: int):
    tokenizer, model = _init_pipeline("all_mpnet_base_v2")

    if len(corpus_sentences) <= 1:
        raise ValueError(
            f"Clusterizer cannot perform with {len(corpus_sentences)} sentences"
        )

    length_sorted_idx = np.argsort(
        [-_get_length(sentence) for sentence in corpus_sentences]
    )
    sentences_sorted = [corpus_sentences[idx] for idx in length_sorted_idx]

    for start_index in trange(0, len(corpus_sentences), batch_size, desc="Batches"):
        sentences_batch = corpus_sentences[start_index : start_index + batch_size]

        features = tokenizer(sentences_batch)


def _get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    device = torch.device("cpu")
    return device


def _init_pipeline(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Parameters = {params}")

    return tokenizer, model


def _get_length(text: Union[str, List[str]]):
    return sum([len(t) for t in text])
