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
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)
        self.base_model._prune_heads(heads_to_prune)

    def get_extended_attention_mask(self, attention_mask, input_shape, device=device):
        """
        Makes broadcastable attention and casual masks so that future and masked tokens are ignored.
        """

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:

            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                casual_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]

                casual_mask = casual_mask.to(attention_mask.dtype)
                extended_attention_mask = casual_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = casual_mask[:, None, :, :]
        else:
            raise ValueError("Wrong shape for input_ids")

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == torch.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError("Not Recognized!!")

        return encoder_extended_attention_mask

