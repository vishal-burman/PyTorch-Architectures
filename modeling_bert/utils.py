from collections import OrderedDict
import torch

def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):

    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

class ModelOutput(OrderedDict):

    def __post_init__(self):
        class_fields = fields(self)

