import fire
from tqdm import trange


def make_batch(corpus_sentences: List[str], batch_size: int):
    if len(corpus_sentences) <= 1:
        raise ValueError(
            f"Clusterizer cannot perform with {len(corpus_sentences)} sentences"
        )

    length_sorted_idx = np.argsort(
        [-_get_length(sentence) for sentence in corpus_sentences]
    )
    sentences_sorted = [corpus_sentences[idx] for idx in length_sorted_idx]

    for start_index in trange(0, len(corpus_sentences), batch_size, desc="Batches"):
        pass


def _get_length(text: Union[str, List[str]]):
    return sum([len(t) for t in text])
