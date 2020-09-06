import inspect
from collections import OrderedDict
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

            # self.config.is_decoder ~ False (default)
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                casual_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]

                casual_mask = casual_mask.to(attention_mask.dtype)
                extended_attention_mask = casual_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                # extended_attention_mask ~ [batch_size, extra, extra, max_seq_len]
                extended_attention_mask = attention_mask[:, None, None, :]
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

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        Prepare the head mask if needed
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            # head_mask = [num_hidden_layers]
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """
        -> [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        """

        if head_mask.dim() == 1:
            # head_mask ~ [1, 1, num_heads, 1, 1] check?
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # head_mask ~ [num_hidden_layers, 1, num_heads, 1, 1]
            # expand used instead of repeat to reduce memory consumption
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        assert head_mask.dim() == 5
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        """
        Functions chunks input_tensors to smaller input_tensor parts of size 'chunk_size'
        over the dimension 'chunk_dim'. It then applies a layer of 'forward_fn' to each
        chunk independently to save memory.
        """

        assert len(input_tensors) > 0
        tensor_shape = input_tensors[0].shape
        assert all(
                input_tensor.shape == tensor_shape for input_tensor in input_tensors
                )

        num_args_in_forward_chunk_fn = len(inpect.signature(forward_fn).parameters)
        assert num_args_in_forward_chunk_fn == len(input_tensors)

        # chunk_size ~ 0
        if chunk_size > 0:
            assert(input_tensors[0].shape[chunk_dim] % chunk_size == 0)

            num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

            # check input tensor into tuples
            input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
            # apply forward fn to every tuple
            output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
            # concatenate output at same dimension
            return torch.cat(output_chunks, dim=chunk_dim)

        return forward_fn(*input_tensors)
