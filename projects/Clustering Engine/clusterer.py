from collections import namedtuple
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


class Clusterer:
    def __init__(self, sentence_encoder: str = "all-MiniLM-L12-v2"):
        self.model = SentenceTransformer(sentence_encoder)
        self.record = namedtuple("record", ["chunk", "chunk_embeds", "chunk_signature"])

    def create_chunks(
        self, sentences: List[str], chunk_size: int, verbose: bool = False
    ) -> List[List[str]]:
        chunks = [
            sentences[i : i + chunk_size] for i in range(0, len(sentences), chunk_size)
        ]

        if verbose:
            print(f"No of windows created from {len(sentences)} sentences: {len(chunks)}")

        return chunks

    def encode_sentences(self, sentences: List[str], verbose: bool = False) -> np.array:
        sentences_embeds = self.model.encode(
            sentences, batch_size=128, show_progress_bar=verbose
        )
        return sentences_embeds

    def create_signature(self, sentences_embeds: np.array, verbose: bool = False):
        if sentences_embeds.ndim == 1:
            sentences_embeds = sentences_embeds.reshape(1, -1)
        sentences_signature = np.mean(sentences_embeds, axis=0, keepdims=True)

        return sentences_signature

    def cluster(
        self, sentences: List[str], chunk_size: int = 10000, verbose: bool = False
    ):  # TODO define a return type
        chunks = self.create_chunks(sentences, chunk_size=chunk_size, verbose=verbose)
        chunks_embeds = [
            self.encode_sentences(chunk, verbose=verbose) for chunk in chunks
        ]
        chunks_signature = [self.create_signature(ce) for ce in chunks_embeds]
        assert len(chunks_embeds) == len(chunks_signature)

        # chunks_vector = list(zip(chunks_embeds, chunks_signature))
        chunks_records = [
            self.record(c, ce, cs)
            for c, ce, cs in zip(chunks, chunks_embeds, chunks_signature)
        ]
        return chunks_records