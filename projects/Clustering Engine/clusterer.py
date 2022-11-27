from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Clusterer:
    def __init__(self, sentence_encoder: str = "all-MiniLM-L12-v2"):
        self.model = SentenceTransformer(sentence_encoder)

    def create_chunks(self, sentences: List[str], chunk_size: int = 10000, verbose: bool = False) -> List[List[str]]:
        chunks = [sentences[i: i + chunk_size]
                  for i in range(0, len(sentences), chunk_size)]

        if verbose:
            print(
                f"No of windows created from {len(sentences)}: {len(chunks)}")

        return chunks

    def encode_sentences(self, sentences: List[str], verbose: bool = False) -> np.array:
        sentences_embeds = self.model.encode(sentences, batch_size=128, show_progress_bar=verbose)
        return sentences_embeds
    
    def create_signature(self, sentences_embeds: np.array):
        if sentences_embeds.ndim == 1:
            sentences_embeds = sentences_embeds.reshape(1, -1)
        sentences_signature = np.mean(sentences_embeds, axis=0, keepdims=True)

        return sentences_signature

    def cluster(self, sentences: List[str]):  # TODO define a return type
        chunks = self.create_chunks(sentences)
        pass
