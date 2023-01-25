from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


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

    def create_cluster_from_chunk(
        self, sentences: List[str], sentences_embeds: np.array, **kwargs
    ):
        cluster_model = AgglomerativeClustering(**kwargs)
        cluster_model.fit(sentences_embeds)
        labels = cluster_model.labels_

        clustered_sentences = [[] for _ in range(max(labels) + 1)]
        for sentence_idx, label in enumerate(labels):
            clustered_sentences[label].append(sentences[sentence_idx])

        return clustered_sentences

    def post_filter_clusters(self, clusters: List[List[str]], min_community_size: int):
        if min_community_size < 1:
            raise RuntimeError(
                f"Minimum cluster size should be greater than or equal to 1"
            )

        clusters = list(filter(lambda x: len(x) >= min_community_size, clusters))
        clusters = sorted(clusters, key=len, reverse=True)

        return clusters

    def get_centroid_embeddings(self, clusters: List[List[str]], verbose: bool):
        centroids = [c[0] for c in clusters]
        print(f"Creating centroids embeddings...")
        centroids_embeds = self.model.encode(centroids, show_progress_bar=verbose)
        return centroids_embeds

    def _merge_clusters(self, clusters, cluster_merge_dict, remaining_idxs):
        pass

    def merge_cluster_with_centroid_similarity(self, clusters: List[List[str]], distance_threshold: float):
        centroid_embeds = self.get_centroid_embeddings(clusters)
        cs_matrix = cosine_similarity(centroid_embeds, centroid_embeds)
        np.fill_diagonal(cs_matrix, 0) # Set the diagonal(all 1) to 0
        cs_matrix = np.triu(cs_matrix)
        rows, columns = np.asarray(cs_matrix >= distance_threshold).nonzero()
        used_idxs, cluster_merge_dict = set(), defaultdict(list)
        for row, column in zip(rows, columns):
            if row not in used_idxs and column not in used_idxs:
                cluster_merge_dict[row].append(column)
                used_idxs.add(column)
        used_idxs = used_idxs.union(set(cluster_merge_dict.keys()))
        remaining_idxs = set(range(len(clusters))) - used_idxs
        new_merged_clusters = self._merge_clusters(clusters, cluster_merge_dict, remaining_idxs)
        assert len(new_merged_clusters) == len(cluster_merge_dict.keys()) + len(remaining_idxs) 
        return new_merged_clusters

    def cluster(
        self,
        sentences: List[str],
        chunk_size: int = 10000,
        normalize_embeddings: bool = True,
        n_clusters: Optional[int] = None,
        affinity: str = "cosine",
        linkage: str = "average",
        distance_threshold: float = 0.4,
        min_community_size: int = 3,
        verbose: bool = False,
    ):  # TODO define a return type
        chunks = self.create_chunks(sentences, chunk_size=chunk_size, verbose=verbose)
        chunks_embeds = [
            self.encode_sentences(chunk, normalize_embeddings, verbose=verbose)
            for chunk in chunks
        ]
        assert len(chunks) == len(chunks_embeds)

        chunks_records = []
        print(f"Preparing clusters from individual chunks...")
        for chunk, chunk_embeds in tqdm(zip(chunks, chunks_embeds), total=len(chunks)):
            chunk_cluster = self.create_cluster_from_chunk(
                chunk,
                chunk_embeds,
                n_clusters=n_clusters,
                affinity=affinity,
                linkage=linkage,
                distance_threshold=distance_threshold,
            )
            cr = ChunkRecord(chunk, chunk_embeds, chunk_cluster)
            chunks_records.append(cr)
        assert len(chunks_records) == len(
            chunks
        ), f"No. of ChunkRecords != No. of chunks"

        all_clusters = []
        for cr in chunks_records:
            all_clusters.extend(cr.chunk_cluster)
        all_clusters = self.post_filter_clusters(all_clusters, min_community_size)

        return all_clusters
