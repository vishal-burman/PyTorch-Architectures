import sys
import pdb
import math
import os
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from activations import gelu, gelu_new, swish

from config_bert import BertConfig
from utils import PretrainedModel, apply_chunking_to_forward 
#from utils import find_pruneable_heads_and_indices

config = BertConfig()

def mish(x):
    """
    Taken from https://arxiv.org/abs/1908.08681 (MISH activation function)

    """
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}

BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Seems a bit hacky (alternative??) 
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            # input_ids ~ [batch_size, seq_max_len]
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # position_ids ~ [1, seq_max_len]
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            # token_type_ids ~ [batch_size, seq_len] 
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # inputs_embeds ~ [batch_size, seq_max_len, emb_size]
            inputs_embeds = self.word_embeddings(input_ids)
        # position_embeds ~ [1, max_seq_len, emb_size]
        position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeds ~ [batch_size, max_seq_len, emb_size]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings ~ [batch_size, max_seq_len, emb_size]
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # embeddings ~ [batch_size, max_seq_len, emb_size]
        embeddings = self.LayerNorm(embeddings)
        # embeddings ~ [batch_size, max_seq_len, emb_size]
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads # self.num_attention_heads ~ 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # self.attention_head_size ~ int(768/12) ~ 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size # self.all_head_size ~ 768
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None): # hidden_states ~ [batch_size, max_len, emb_size] && attention_mask ~ [batch_size, 1, 1, seq_len]
        bs, slen = hidden_states.shape[:2]
        mixed_query_layer = self.query(hidden_states) # mixed_query_layer ~ [batch_size, max_seq_len, emb_size] 
        mixed_key_layer = self.key(hidden_states) # mixed_key_layer ~ [batch_size, max_seq_len, emb_size]
        mixed_value_layer = self.value(hidden_states) # mixed_value_layer ~ [batch_size, max_seq_len, emb_size]
        query_layer = mixed_query_layer.view(bs, slen, self.num_attention_heads, -1).permute(0, 2, 1, 3) # query_layer ~ [batch_size, n_heads, max_len, emb_dim//n_heads]
        key_layer = mixed_key_layer.view(bs, slen, self.num_attention_heads, -1).permute(0, 2, 1, 3) # key_layer ~ [batch_size, n_heads, max_len, emb_dim//n_heads]
        value_layer = mixed_value_layer.view(bs, slen, self.num_attention_heads, -1).permute(0, 2, 1, 3) # value_layer ~ [batch_size, n_heads, max_len, emb_dim//n_heads]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # attention_scores ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # attention_scires ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
        attention_scores = attention_scores + attention_mask # attention_scores ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # attention_probs ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
        attention_probs = self.dropout(attention_probs) # attention_probs ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
        context_layer = torch.matmul(attention_probs, value_layer) # context_layer ~ [batch_size, num_attention_heads, max_seq_len, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # context_layer ~ [batch_size, max_seq_len, num_attention_heads, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # new_context_layer_shape ~ [batch_size, max_seq_len, emb_size]
        context_layer = context_layer.view(*new_context_layer_shape) # context_layer ~ [batch_size, max_seq_len, emb_dim]
        outputs = (context_layer, attention_probs) if output_attention else (context_layer,)
        return outputs # outputs ~ ([batch_size, max_seq_len, all_head_size])

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        hidden_states = self.dense(hidden_states)
        # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        hidden_states = self.dropout(hidden_states)
        # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,
            hidden_states, # hidden_states ~ [batch_size, max_seq_len, hidden_size]
            attention_mask=None, # attention_mask ~ [batch_size, max_seq_len, hidden_size]
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,):

        
        self.outputs = self.self(hidden_states, attention_mask) # self.outputs ~ ([batch_size, max_seq_len, emb_dim])
        # attention_output ~ [batch_size, max_seq_len, emb_size]
        attention_output = self.output(self.outputs[0], hidden_states)
        # outputs ~ ([batch_size, max_seq_len, hidden_size])
        outputs = (attention_output,) + self.outputs[1:]
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            # config.hidden_act ~ 'gelu'
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states): # hidden_states ~ [batch_size, max_seq_len, all_head_size]
        # hidden_states ~ [batch_size, max_seq_len, intermediate_size] where intermediate_size = 3072
        hidden_states = self.dense(hidden_states)
        # hidden_states ~ [batch_size, max_seq_len, intermediate_size]
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor): # hidden_states ~ [batch_size, max_seq_len, intermediate_size] & input_tensor ~ [batch_size, max_seq_len, all_head_size]
        # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        hidden_states = self.dense(hidden_states)
        # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        hidden_states = self.dropout(hidden_states)
        # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None): # hidden_states ~ [batch_size, max_len, emb_dim] && attention_mask ~ [batch_size, 1, 1, max_len]
        self_attention_outputs = self.attention(hidden_states, attention_mask) # self_attention_outputs ~ ([batch_size, max_seq_len, all_head_size]) where all_head_size = hidden_size
        attention_output = self_attention_outputs[0] # attention_output ~ [batch_size, max_seq_len, all_head_size]
        outputs = self_attention_outputs[1:] # outputs ~ [max_seq_len, all_head_size]
        layer_output = self.feed_forward_chunk(attention_output) # layer_output ~ [batch_size, max_seq_len, hidden_size]
        outputs = (layer_output,) + outputs # outputs ~ ([batch_size, max_seq_len, hidden_size])
        return outputs

    def feed_forward_chunk(self, attention_output): # attention_output ~ [batch_size, max_seq_len, all_head_size]
        intermediate_output = self.intermediate(attention_output) # intermediate_output ~ [batch_size, max_seq_len, intermediate_size]
        layer_output = self.output(intermediate_output, attention_output) # layer_output ~ [batch_size, max_seq_len, hidden_size]
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states,  attention_mask=None) # hidden_states ~ [batch_size, max_len, emb_dim] #attention_mask ~ [batch_size, 1, 1, seq_len])
        for i, layer_module in enumerate(self.layer): # self.layer ~ BertLayer 
            layer_outputs = layer_module(hidden_states, attention_mask)
            
            hidden_states = layer_outputs[0] # hidden_states ~ [batch_size, max_seq_len, emb_size]
        return tuple(v for v in [hidden_states] if v is not None)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0] # first_token_tensor ~ [batch_size, emb_size]
        pooled_output = self.dense(first_token_tensor) # pooled_output ~ [batch_size, emb_size]
        pooled_output = self.activation(pooled_output) # pooled_output ~ [batch_size, emb_size]
        return pooled_output

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids=None, attention_mask=None):
        input_shape = input_ids.size()
        extended_attention_mask = attention_mask[input_shape[0], None, None, input_shape[1]] # extended_attention_mask ~ [batch_size, extra, extra, max_seq_len]
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids) # embedding_output ~ [batch_size, max_seq_len, emb_size]
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask) #encoder_outputs ~ ([batch_size, max_len, emb_dim])
        sequence_output = encoder_outputs[0] #sequence_output ~ [batch_size, max_seq_len, emb_size]
        pooled_output = self.pooler(sequence_output) # pooled_output ~ [batch_size, emb_size]
        return (sequence_output, pooled_output) + encoder_outputs[1:]  # return ~ ([batch_size, max_seq_len, emb_size], [batch_size, emb_size])

class BertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids = None, attention_mask = None, labels = None):
        outputs = self.bert(input_ids, attention_mask=attention_mask) # outputs ~ ([batch_size, max_seq_len, hidden_size], [batch_size, hidden_size])
        pooled_output = outputs[1] # pooled_output ~ [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output) # pooled_output ~ [batch_size, hidden_size]
        logits = self.classifier(pooled_output) # logits ~ [batch_size, num_labels]
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        output = (logits,)
        return ((loss,) + output) if loss is not None else output  # return ~ (loss, [batch_size, num_labels])
