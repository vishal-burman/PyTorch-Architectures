from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_with_tokens(t, token_ids): # t ~ [batch_size, max_len] | token_ids ~ [num_ignore_token_ids]
    init_no_mask = torch.full_like(t, False, dtype=torch.bool) # init_no_mask ~ [batch_size, max_len]
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask) # mask ~ [batch_size, max_len]
    return mask

class MLM(nn.Module):
    def __init__(self, pad_token_id, mask_token_id, mask_ignore_token_ids=[]):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, input_ids): # input_ids ~ [batch_size, max_len]
        no_mask = mask_with_tokens(input_ids, self.mask_ignore_token_ids) # no_mask ~ [batch_size, max_len]
