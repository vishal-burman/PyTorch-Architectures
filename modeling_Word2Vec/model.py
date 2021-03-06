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
        self.W = nn.Linear(vocab_size, emb_size, bias=False)
        self.WT = nn.Linear(emb_size, vocab_size, bias=False)

    def forward(self, X): # X ~ [bs, vocab_size]
        hidden_layer = self.W(X) # hidden_layer ~ [bs, emb_size]
        output_layer = self.WT(hidden_layer) # output_layer ~ [bs, vocab_size]
        return output_layer
