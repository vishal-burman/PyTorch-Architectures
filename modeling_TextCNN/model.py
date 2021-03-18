import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, num_filters, filter_sizes, vocab_size, embedding_size, sequence_length, num_classes=2):
        super().__init__()
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.W = nn.Embedding(self.vocab_size, self.embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, self.num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([self.num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, self.num_filters, (size, self.embedding_size)) for size in self.filter_sizes])

    def forward(self, x): # x ~ [batch_size, max_seq_len]
        x = self.W(x) # x ~ [batch_size, max_seq_len, embedding_size]
        x = x.unsqueeze(1) # x ~ [batch_size, 1, max_seq_len, embedding_size]
        
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x)) # h ~ [batch_size, num_filters, self.filter_size, 1] 
            mp = nn.MaxPool2d((self.sequence_length - self.filter_sizes[i] + 1, 1))
            pooled = mp(h).permute(0, 3, 2, 1) # pooled ~ [batch_size, 1, 1, 3]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes)) # h_pool ~ [batch_size, 1, 1, 3 * 3]
        h_pool_flat = h_pool.reshape(h_pool.size(0), -1) # h_pool_flat ~ [batch_size, 1 * 1 * 3 * 3]
        logits = self.Weight(h_pool_flat) + self.Bias # logits ~ [batch_size, num_classes]
        return logits
        """
