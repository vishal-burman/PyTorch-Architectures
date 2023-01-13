import argparse
from typing import List

from clusterer import Clusterer


def filter_sentences(sentences: List[str]) -> List[str]:
    sentences = list(
        filter(lambda x: len(x.split()) > 2 and len(x.split()) < 128, sentences)
    )
    return sentences


def read_file(filename: str) -> List[str]:
    file_ = open(filename, "r").readlines()
    file_ = list(map(lambda x: x.strip(), file_))
    file_ = list(set(file_))
    return file_


def write_cluster_file(clusters: List[str], filename: str):
    with open(filename, "w") as fout:
        for cluster_idx, cluster in enumerate(clusters):
            fout.write(f"Cluster {cluster_idx}:\n")
            for sentence in cluster:
                fout.write(sentence + "\n")
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    parser.add_argument("-m", "--model", default="all-MiniLM-L12-v2")
    parser.add_argument("-cs", "--chunk_size", default=1000)

    args = parser.parse_args()
    filename = args.filename
    sentence_encoder = args.model
    chunk_size = args.chunk_size

    clusterer = Clusterer(sentence_encoder)

    list_sentences = read_file(filename)
    list_sentences = filter_sentences(list_sentences)
    print(f"Total read sentences(filtered): {len(list_sentences)}")

    all_clusters = clusterer.cluster(
        list_sentences, chunk_size=chunk_size, verbose=True
    )
    write_cluster_file(clusters=all_clusters, filename=f"{filename}_clusters")
