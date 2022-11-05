import argparse
from typing import List
from clusterer import Clusterer


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
