import sys
import math
import os
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from activations import gelu, gelu_new, swish

from config_bert import BertConfig
from utils import PretrainedModel
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

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # TODO
            print("Error!!!")
        # self.num_attention_heads ~ 12
        self.num_attention_heads = config.num_attention_heads
        # self.attention_head_size ~ int(768/12) ~ 64
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # self.all_head_size ~ 768
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # x ~ [batch_size, max_seq_len, emb_size]
        # new_x_shape ~ [batch_size, max_seq_len, num_attention_heads, attention_head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # num_attention_heads * attention_head_size = emb_size
        # x ~ [batch_size, max_seq_len, num_attention_heads, attention_head_size]
        x = x.view(*new_x_shape)
        # x.permute ~ [batch_size, num_attention_heads, max_seq_len, attention_head_size]
        return x.permute(0, 2, 1, 3)

    def forward(self, 
            hidden_states, # hidden_states ~ [batch_size, max_seq_len, hidden_size] 
            attention_mask=None, #attention_mask ~ [batch_size, max_seq_len, hidden_size] 
            head_mask=None, 
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attention=False):

        # mixed_query_layer ~ [batch_size, max_seq_len, emb_size] where emb_size = self.all_head_size
        mixed_query_layer = self.query(hidden_states)

        # encoder_hidden_states ~ None
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            # mixed_key_layer ~ [batch_size, max_seq_len, emb_size]
            mixed_key_layer = self.key(hidden_states)
            # mixed_value_layer ~ [batch_size, max_seq_len, emb_size]
            mixed_value_layer = self.value(hidden_states)

        # query_layer ~ [batch_size, num_attention_heads, max_seq_len, attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # key_layer ~ [batch_size, num_attention_heads, max_seq_len, attention_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # value_layer ~ [batch_size, num_attention_heads, max_seq_len, attention_head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # attention_scores ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scires ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # attention_scores ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
            attention_scores = attention_scores + attention_mask

        # attention_probs ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs ~ [batch_size, num_attention_heads, max_seq_len, max_seq_len]
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context_layer ~ [batch_size, num_attention_heads, max_seq_len, attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)

        # context_layer ~ [batch_size, max_seq_len, num_attention_heads, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # new_context_layer_shape ~ [batch_size, max_seq_len, all_head_size] where all_head_size = emb_size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        # context_layer ~ [batch_size, max_seq_len, all_head_size]
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attention else (context_layer,)
        
        # outputs ~ ([batch_size, max_seq_len, all_head_size])
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)

        # Prune Linear Layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self,
            hidden_states, # hidden_states ~ [batch_size, max_seq_len, hidden_size]
            attention_mask=None, # attention_mask ~ [batch_size, max_seq_len, hidden_size]
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,):

        # self.outputs ~ ([batch_size, max_seq_len, hidden_size])
        self.outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions)
        attention_output = self.output(self.outputs[0], hidden_states)
        outputs = (attention_output) + self.outputs[1:]
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
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self,
            hidden_states, # hidden_states ~ [batch_size, max_seq_len, hidden_size]
            attention_mask=None, # [batch_size, max_seq_len, hidden_size]
            head_mask=None, 
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False):

        # self_attention_outputs ~ ([batch_size, max_seq_len, all_head_size]) where all_head_size = hidden_size
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        # attention_output ~ [batch_size, max_seq_len, all_head_size]
        attention_output = self_attention_outputs[0]
        # outputs ~ [max_seq_len, all_head_size]
        outputs = self_attention_outputs[1:]

        # self.is_decoder ~ False encoder_hidden_states ~ None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(self, 'crossattention'), f"some statement"
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        # layer_output ~ [batch_size, max_seq_len, hidden_size]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output): # attention_output ~ [batch_size, max_seq_len, all_head_size]
        # intermediate_output ~ [batch_size, max_seq_len, intermediate_size] where intermediate_size = 3072 TODO
        intermediate_output = self.intermediate(attention_output)
        # layer_output ~ [batch_size, max_seq_len, hidden_size] #TODO
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
            hidden_states, # hidden_states ~ [batch_size, max_seq_len, hidden_size]
            attention_mask=None, # attention_mask ~ [batch_size, max_seq_len, hidden_size]
            head_mask=None, # head_mask ~ [None] * num_hidden_layers
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False):

        # all_hidden_states ~ False
        all_hidden_states = () if output_hidden_states else None
        # all_attentions ~ False
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer): # self.layer ~ BertLayer
            # output_hidden_states ~ False
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # gradient_checkpointing ~ False
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states, # [batch_size, max_seq_len, emb_size]
                        attention_mask, # [batch_size, extra, extra, max_seq_len]
                        head_mask[i], # head_mask ~ [None] * num_hidden_layers
                        encoder_hidden_states, # None
                        encoder_attention_mask, # None
                        )
            else:
                layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        head_mask[i],
                        encoder_hidden_states,
                        encoder_attention_mask,
                        output_attentions,
                        )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=all_hidden_states, attention=all_attentions
                )

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    def  __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertPreTrainedModel(PretrainedModel):
    # TODO
    config_class = BertConfig
    #load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        # TODO Not needed for sequence classification
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model"""
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):

        # output_attentions ~ False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states ~ False
        output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
        # return_dict ~ False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and input_embeds is not None:
            raise ValueError("Both input_ids and input_embeds cannot exist at the same time")
        elif input_ids is not None:
            # input_shape ~ (batch_size, max_seq_len)
            input_shape = input_ids.size()
        elif input_embeds is not None:
            input_shape = input_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or input_embeds")

        device = input_ids.device if input_ids is not None else input_embeds.device

        if attention_mask is None:
            # attention_mask ~ [batch_size, max_seq_len]
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            # token_type_ids ~ [batch_size, max_seq_len]
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # extended_attention_mask ~ [batch_size, extra, extra, max_seq_len]
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # self.config.is_decoder ~ False & encoder_hidden_states ~ None
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # head_mask ~ [None] * num_hidden_layers
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # embedding_output ~ [batch_size, max_seq_len, emb_size]
        embedding_output = self.embeddings(
                input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, input_embeds=input_embeds
                )

        encoder_outputs = self.encoder(
                embedding_output, # [batch_size, max_seq_len, emb_size]
                attention_mask=extended_attention_mask, # [batch_size, extra, extra, max_seq_len]
                head_mask=head_mask, # [None] * num_hidden_layers
                encoder_hidden_states=encoder_hidden_states, # None
                encoder_attention_mask=encoder_extended_attention_mask, # None
                output_attentions=output_attentions, # False
                output_hidden_states=output_hidden_states, # False
                return_dict=return_dict,
                )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                )



class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            input_embeds = None,
            labels = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                input_embeds=input_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )

        pooled_output = outputs[1]
        pooled_output = self.dropout(outputs)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
                )
