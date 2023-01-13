import argparse
from typing import List

from clusterer import Clusterer


def filter_sentences(sentences: List[str]) -> List[str]:
    sentences = list(filter(lambda x: len(x.split()) > 2 and len(x.split()) < 128))
    return sentences


def read_file(filename: str) -> List[str]:
    file_ = open(filename, "r").readlines()
    file_ = list(map(lambda x: x.strip(), file_))
    file_ = list(set(file_))
    return file_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    parser.add_argument("-m", "--model", default="all-MiniLM-L12-v2")

    args = parser.parse_args()
    filename = args.filename
    sentence_encoder = args.model

    clusterer = Clusterer(sentence_encoder)

    list_sentences = read_file(filename)
    list_sentences = filter_sentences(list_sentences)
