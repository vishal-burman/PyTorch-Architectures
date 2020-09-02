from collections import OrderedDict
import torch
import torch.nn as nn

def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):

    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

class PretrainedModel(nn.Module):
    config_class = None
    base_model_prefix = ""
    authorized_missing_keys = None

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()

        self.config = config

    def init_weights(self):
        """
        Initializes and prunes weights if needed
        """

        self.apply(self._init_weights)

        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        # Tie weights if needed
        self.tie_weights()

    def prune_heads(self, heads_to_prune):
        """
        Prunes heads of the base model
        """

        for layer, heads in heads_to_prune.items():
            pass
        pass
