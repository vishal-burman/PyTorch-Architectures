from functools import reduce
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_with_tokens(t, token_ids): # t ~ [batch_size, max_len] | token_ids ~ [num_ignore_token_ids]
    init_no_mask = torch.full_like(t, False, dtype=torch.bool) # init_no_mask ~ [batch_size, max_len]
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask) # mask ~ [batch_size, max_len]
    return mask

def get_mask_subset_with_prob(mask, prob): # mask ~ [batch_size, seq_len]
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len) 
    num_tokens = mask.sum(dim=-1, keepdim=True) # num_tokens ~ [batch_size, 1]
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil()) # max_excess ~ [batch_size, max_len]
    mask_excess = mask_excess[:, :max_masked] # mask_excess ~ [batch_size, max_masked]
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, 1e-9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

class MLM(nn.Module):
    def __init__(self, mask_prob=0.15, pad_token_id=0, mask_token_id=2, mask_ignore_token_ids=[]):
        super().__init__()
        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, input_ids): # input_ids ~ [batch_size, max_len]
        no_mask = mask_with_tokens(input_ids, self.mask_ignore_token_ids) # no_mask ~ [batch_size, max_len]
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)
        
