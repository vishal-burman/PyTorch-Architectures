from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


@dataclass
class ChunkRecord:
    chunk: List[str]
    chunk_embeds: np.array
    chunk_cluster: List[List[str]]


class Clusterer:
    def __init__(self, sentence_encoder: str = "all-MiniLM-L12-v2"):
        self.model = SentenceTransformer(sentence_encoder)

    def create_chunks(
        self, sentences: List[str], chunk_size: int, verbose: bool = False
    ) -> List[List[str]]:
        chunks = [
            sentences[i : i + chunk_size] for i in range(0, len(sentences), chunk_size)
        ]

        if verbose:
            print(
                f"No of windows created from {len(sentences)} sentences: {len(chunks)}"
            )

        return chunks

    def normalize_embeddings(self, sentences_embeds: np.array) -> np.array:
        sentences_embeds = sentences_embeds / np.linalg.norm(
            sentences_embeds, axis=1, keepdims=True
        )
        return sentences_embeds

    def encode_sentences(
        self,
        sentences: List[str],
        normalize_embeddings: bool,
        verbose: bool = False,
    ) -> np.array:
        sentences_embeds = self.model.encode(
            sentences, batch_size=128, show_progress_bar=verbose
        )
        if normalize_embeddings:
            sentences_embeds = self.normalize_embeddings(sentences_embeds)

        return sentences_embeds

    def create_signature(self, sentences_embeds: np.array, verbose: bool = False):
        if sentences_embeds.ndim == 1:
            sentences_embeds = sentences_embeds.reshape(1, -1)
        sentences_signature = np.mean(sentences_embeds, axis=0, keepdims=True)

        return sentences_signature

    def create_cluster_from_chunk(self, sentences: List[str], sentences_embeds: np.array, **kwargs):
        cluster_model = AgglomerativeClustering(**kwargs)
        cluster_model.fit(sentences_embeds)
        labels = cluster_model.labels_

        clustered_sentences = [[] for _ in range(max(labels) + 1)]
        for sentence_idx, label in enumerate(labels):
            clustered_sentences[label].append(sentences[sentence_idx])
        
        return clustered_sentences

    def cluster(
        self,
        sentences: List[str],
        chunk_size: int = 10000,
        normalize_embeddings: bool = True,
        n_clusters: Optional[int] = None,
        affinity: str = "cosine",
        linkage: str = "average",
        distance_threshold: float = 0.4,
        verbose: bool = False,
    ):  # TODO define a return type
        chunks = self.create_chunks(sentences, chunk_size=chunk_size, verbose=verbose)
        chunks_embeds = [
            self.encode_sentences(chunk, normalize_embeddings, verbose=verbose)
            for chunk in chunks
        ]
        assert len(chunks) == len(chunks_embeds)

        chunks_records = []
        for chunk, chunk_embeds in zip(chunks, chunks_embeds):
            cluster_chunk = self.create_cluster_from_chunk(chunk, chunk_embeds, n_clusters=n_clusters, affinity=affinity, linkage=linkage, distance_threshold=distance_threshold)
            cr = ChunkRecord(chunk, chunk_embeds, cluster_chunk)
            chunks_records.append(cr)
        assert len(chunks_records) == len(chunks), f"#ChunkRecords != #chunks"

        return chunks_records
