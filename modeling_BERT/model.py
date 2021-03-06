import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from config_bert import BertConfig
from utils import gelu_new 
config = BertConfig()
BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None):
        input_shape = input_ids.size() # input_ids ~ [batch_size, seq_max_len]
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length] # position_ids ~ [1, seq_max_len]
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device) # token_type_ids ~ [batch_size, seq_len]
        inputs_embeds = self.word_embeddings(input_ids) # inputs_embeds ~ [batch_size, seq_max_len, emb_size]
        position_embeddings = self.position_embeddings(position_ids) # position_embeds ~ [1, max_seq_len, emb_size]
        token_type_embeddings = self.token_type_embeddings(token_type_ids) # token_type_embeds ~ [batch_size, max_seq_len, emb_size]
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings # embeddings ~ [batch_size, max_seq_len, emb_size]
        embeddings = self.LayerNorm(embeddings) # embeddings ~ [batch_size, max_seq_len, emb_size]
        embeddings = self.dropout(embeddings) # embeddings ~ [batch_size, max_seq_len, emb_size]
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
        outputs = (context_layer,)
        return outputs # outputs ~ ([batch_size, max_seq_len, all_head_size])

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states)) # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None): # hidden_states ~ [batch_size, max_len, emb_size] && attention_mask ~ [batch_size, 1, 1, max_len]
        self.outputs = self.self(hidden_states, attention_mask) # self.outputs ~ ([batch_size, max_seq_len, emb_dim])
        attention_output = self.output(self.outputs[0], hidden_states) # attention_output ~ [batch_size, max_seq_len, emb_size]
        outputs = (attention_output,) + self.outputs[1:] # outputs ~ ([batch_size, max_seq_len, hidden_size])
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu_new 

    def forward(self, hidden_states): # hidden_states ~ [batch_size, max_seq_len, emb_size]
        hidden_states = self.intermediate_act_fn(self.dense(hidden_states)) # hidden_states ~ [batch_size, max_len, intermediate_size]
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor): # hidden_states ~ [batch_size, max_seq_len, intermediate_size] & input_tensor ~ [batch_size, max_seq_len, emb_dim]
        hidden_states = self.dense(hidden_states) # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        hidden_states = self.dropout(hidden_states) # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # hidden_states ~ [batch_size, max_seq_len, hidden_size]
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None): # hidden_states ~ [batch_size, max_len, emb_dim] && attention_mask ~ [batch_size, 1, 1, max_len]
        self_attention_outputs = self.attention(hidden_states, attention_mask) # self_attention_outputs ~ ([batch_size, max_seq_len, emb_dim]) 
        attention_output = self_attention_outputs[0] # attention_output ~ [batch_size, max_seq_len, emb_dim]
        outputs = self_attention_outputs[1:] # outputs ~ [max_seq_len, emb_dim]
        layer_output = self.output(self.intermediate(attention_output), attention_output) # layer_output ~ [batch_size, max_seq_len, emb_dim]
        outputs = (layer_output,) + outputs # outputs ~ ([batch_size, max_seq_len, hidden_size])
        return outputs

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states,  attention_mask=None): # hidden_states ~ [batch_size, max_len, emb_dim] #attention_mask ~ [batch_size, 1, 1, seq_len])
        for i, layer_module in enumerate(self.layer): # self.layer ~ BertLayer 
            hidden_states = layer_module(hidden_states, attention_mask)[0]  # hidden_states ~ [batch_size, max_seq_len, emb_size]
        return tuple(v for v in [hidden_states] if v is not None)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0])) # return ~ [batch_size, emb_size]

class BertClassify(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        extended_attention_mask = attention_mask[:, None, None, :] # extended_attention_mask ~ [batch_size, extra, extra, max_seq_len]
        embedding_output = self.embeddings(input_ids=input_ids) # embedding_output ~ [batch_size, max_seq_len, emb_size]
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask) #encoder_outputs ~ ([batch_size, max_len, emb_dim])
        sequence_output = encoder_outputs[0] #sequence_output ~ [batch_size, max_seq_len, emb_size]
        pooled_output = self.dropout(self.pooler(sequence_output)) # pooled_output ~ [batch_size, emb_size]
        logits = self.classifier(pooled_output) # logits ~ [batch_size, num_labels]
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        output = (logits,)
        return ((loss,) + output) if loss is not None else output  # return ~ (loss, [batch_size, num_labels])
