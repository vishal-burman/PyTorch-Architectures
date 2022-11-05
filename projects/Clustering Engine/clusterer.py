import argparse
from typing import List

def read_file(filename: str) -> List[str]:
    file_ = open(filename, "r").readlines()
    file_ = list(map(lambda x: x.strip(), file_))
    file_ = list(set(file_))
    return file_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")

    args = parser.parse_args()
    filename = args.filename

    list_sentences = read_file(filename)