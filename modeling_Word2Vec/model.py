import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, emb_size):
        """
        Arguments:
        vocab_size --> Number of vocabulary words
        emb_size --> Embedding dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.W = nn.Embedding(vocab_size, emb_size)
        self.WT = nn.Linear(emb_size, vocab_size)

    def forward(self):
        pass
