from typing import List

from sentence_transformers import SentenceTransformer


class Clusterer:
    def __init__(self, sentence_encoder: str = "all-MiniLM-L12-v2"):
        self.model = SentenceTransformer(sentence_encoder)

    def create_chunks(self, sentences: List[str], chunk_size: int = 10000, verbose: bool = False):
        chunks = [sentences[i: i + chunk_size]
                   for i in range(0, len(sentences), chunk_size)]

        if verbose:
            print(
                f"No of windows created from {len(sentences)}: {len(chunks)}")

        return chunks

    def cluster(self, sentences: List[str]):  # TODO define a return type
        chunks = self.create_chunks(sentences)
        pass
