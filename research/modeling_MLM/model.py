from functools import reduce
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids): # t ~ [batch_size, max_len] | token_ids ~ [num_ignore_token_ids]
    init_no_mask = torch.full_like(t, False, dtype=torch.bool) # init_no_mask ~ [batch_size, max_len]
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask) # mask ~ [batch_size, max_len]
    return mask

def get_mask_subset_with_prob(mask, prob): # mask ~ [batch_size, max_len]
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len) 
    num_tokens = mask.sum(dim=-1, keepdim=True) # num_tokens ~ [batch_size, 1]
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil()) # max_excess ~ [batch_size, max_len]
    mask_excess = mask_excess[:, :max_masked] # mask_excess ~ [batch_size, max_masked]
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, 1e-9) # rand ~ [batch_size, max_len]
    _, sampled_indices = rand.topk(max_masked, dim=-1) # sampled_indices ~ [batch_size, max_masked]
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0) # sampled_indices ~ [batch_size, max_masked]
    new_mask = torch.zeros((batch, seq_len + 1), device=device) # new_mask ~ [batch_size, max_len + 1]
    new_mask.scatter_(-1, sampled_indices, 1) # new_mask ~ [batch_size, max_len + 1]
    return new_mask[:, 1:].bool() # new_mask ~ [batch_size, max_len]

class MLM(nn.Module):
    def __init__(self, transformer, mask_prob=0.15, pad_token_id=0, mask_token_id=2, num_tokens=None, replace_prob=0.9, mask_ignore_token_ids=[]):
        super().__init__()
        self.transformer = transformer
        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.num_tokens = num_tokens
        self.replace_prob = replace_prob
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, input_ids, **kwargs): # input_ids ~ [batch_size, max_len]
        no_mask = mask_with_tokens(input_ids, self.mask_ignore_token_ids) # no_mask ~ [batch_size, max_len]
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob) # mask ~ [batch_size, max_len]
        mask_indices = torch.nonzero(mask, as_tuple=True) # mask_indices ~ [mask_prob * max_len]
        masked_input = input_ids.clone().detach() # masked_input ~ [batch_size, max_len]
        replace_prob = prob_mask_like(input_ids, self.replace_prob) # replace_prob ~ [batch_size, max_len]
        masked_input = masked_input.masked_fill(mask * replace_prob, self.mask_token_id) # masked_input ~ [batch_size, max_len]
        labels = input_ids.masked_fill(~mask, self.pad_token_id) # labels ~ [batch_size, max_len]
        logits = self.transformer(masked_input, **kwargs).last_hidden_state
        return logits, labels
