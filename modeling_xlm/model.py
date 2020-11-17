import pdb
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from config_xlm import XLMConfig

config = XLMConfig()
gelu = F.gelu

class MultiHeadAttention(nn.Module):
    
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

    def forward(
            self, 
            input,  # input ~ [batch_size, max_len, emb_size]
            mask,  # mask ~ [batch_size, max_len]
            ):
        
        bs, qlen, dim = input.size()
        
        klen = qlen
        
        # n_heads ~ 8 
        n_heads = self.n_heads
        # dim_per_head ~ 1024 // 8 --> 128
        dim_per_head = self.dim // n_heads
        # mask.dim() = 2 --> mask_reshape ~ (bs, 1, 1, klen) where klen = max_len
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """ projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        # q ~ [batch_size, n_heads, max_len, dim_per_head]
        q = shape(self.q_lin(input))
        
        # k ~ [batch_size, n_heads, max_len, dim_per_head] 
        k = shape(self.k_lin(input))
        
        # v ~ [batch_size, n_heads, max_len, dim_per_head] 
        v = shape(self.v_lin(input)) 

        # q ~ [batch_size, n_heads, max_len, dim_per_head] 
        q = q / math.sqrt(dim_per_head)
        
        # q ~ [batch_size, n_heads, max_len, dim_per_head]
        # k.T(2, 3) ~ [batch_size, n_heads, dim_per_head, max_len]
        # scores ~ [batch_size, n_heads, max_len, max_len]
        scores = torch.matmul(q, k.transpose(2, 3))
        
        # mask ~ [batch_size, n_heads, max_len, max_len]
        mask = (mask == 0).view(mask_reshape).expand_as(scores)
        scores.masked_fill_(mask, -float("inf"))

        # weights ~ [batch_size, n_heads, max_len, max_len]
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        # weights ~ [batch_size, n_heads, max_len, max_len]
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        # weights ~ [batch_size, n_heads, max_len, max_len] | v ~ [batch_size, n_heads, max_len, dim_per_head]
        # context ~ [batch_size, n_heads, max_len, dim_per_head]
        context = torch.matmul(weights, v)
        # context = [batch_size, 2, n_heads * dim_per_head]
        context = unshape(context)

        # outputs ~ ([batch_size, max_len, emb_dim])
        outputs = (self.out_lin(context),)
        
        # outputs ~ ([batch_size, max_len, emb_size])
        return outputs

class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super().__init__()
        self.dropout = config.dropout
        self.lin_1 = nn.Linear(in_dim, dim_hidden)
        self.lin_2 = nn.Linear(dim_hidden, out_dim)
        self.act = gelu if config.gelu_activation else F.relu
        self.seq_len_dim = 1

    def forward(
            self, 
            input, # input ~ [batch_size, max_len, emb_size]
            ):
        # x ~ [batch_size, max_len, dim_hidden] where dim_hidden = emb_dim * 4
        x = self.lin_1(input)
        # x ~ [batch_size, max_len, dim_hidden]
        x = self.act(x)
        # x ~ [batch_size, max_len, emb_dim]
        x = self.lin_2(x)
        # x ~ [batch_size, max_len, emb_dim]
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x 


class XLMModel(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.n_words = config.vocab_size
        self.pad_index = config.pad_index
        self.dim = config.emb_dim 
        self.hidden_dim = self.dim * 4 
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.dim)
        self.embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=config.layer_norm_eps)
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, config=config))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))
            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, config=config))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))

        self.init_weights()
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand(1, -1))

    def forward(self, input_ids=None, attention_mask=None): # input_ids, attention_mask ~ [batch_size, max_len]
        bs, slen = input_ids.size() # bs ~ batch_size || slen ~ max_len
        lengths = (input_ids != self.pad_index).sum(dim=1).long() # lengths ~ [max_len - (count of pad_tokens)]  & len(lengths) ~ batch_size
        mask, attn_mask = attention_mask, attention_mask # mask, attn_mask ~ [batch_size, max_len] 
        position_ids = self.position_ids[:, :slen] # position_ids ~ [batch_size, max_len]
        inputs_embeds = self.embeddings(input_ids) # inputs_embeds ~ [batch_size, max_len, emb_dim]
        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds) # tensor ~ [batch_size, max_len, emb_dim]
        tensor = self.layer_norm_emb(tensor) # tensor ~ [batch_size, max_len, emb_dim]
        tensor = F.dropout(tensor, p=self.dropout, training=self.training) # tensor ~ [batch_size, max_len, emb_dim]
        tensor *= mask.unsqueeze(-1).to(tensor.dtype) # tensor ~ [batch_size, max_len, emb_dim]
        for i in range(self.n_layers): 
            attn_outputs = self.attentions[i](tensor,  attn_mask) # attn_outputs ~ ([batch_size, max_len, emb_size])
            attn = attn_outputs[0] # attn ~ [batch_size, max_len, emb_size]
            attn = F.dropout(attn, p=self.dropout, training=self.training) # attn ~ [batch_size, max_len, emb_size]
            tensor = tensor + attn # tensor ~ [batch_size, max_len, emb_size]
            tensor = self.layer_norm1[i](tensor) # tensor ~ [batch_size, max_len, emb_size]
            tensor = tensor + self.ffns[i](tensor) # tensor ~ [batch_size, max_len, emb_size]
            tensor = self.layer_norm2[i](tensor) # tensor ~ [batch_size, max_len, emb_size]
            tensor *= mask.unsqueeze(-1).to(dtype=tensor.dtype) # tensor ~ [batch_size, max_len, emb_size]
        return tuple(v for v in [tensor] if v is not None)

class XLMPredLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_words = config.vocab_size
        self.pad_index = config.pad_index
        dim = config.emb_dim
        self.proj = nn.Linear(dim, self.n_words, bias=True)

    def forward(self, x, y=None): #x ~ [batch_size, max_len, emb_size] || y ~ [batch_size, max_len]
        outputs = ()
        scores = self.proj(x) # scores ~ [batch_size, max_len, vocab_size]
        outputs = (scores,) + outputs # outputs ~ ([batch_size, max_len, vocab_size])
        if y is not None:
            loss = F.cross_entropy(scores.view(-1, self.n_words), y.view(-1), reduction='elementwise_mean')
            outputs = (loss,) + outputs
        return outputs # outputs ~ (loss, [batch_size, max_len, vocab_size])

class XLMWithLMHeadModel(XLMPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = XLMModel(config)
        self.pred_layer = XLMPredLayer(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None): #input_ids, attention_mask ~ [batch_size, max_len] || label ~ [batch_size]
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask) # transformer_outputs ~ ([batch_size, max_len, emb_size])
        output = transformer_outputs[0] # output ~ [batch_size, max_len, emb_size]
        outputs = self.pred_layer(output, labels) # outputs ~ (loss, [batch_size, max_len, vocab_size])
        return outputs
