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

class XLMPretrainedModel(nn.Module):
    config_class = XLMConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            if self.config is not None and self.config.embed_init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.embed_init_std)
        
        if isinstance(module, nn.Linear):
            if self.config is not None and self.config.init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.init_std)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class XLMModel(XLMPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # dictionary / languages
        self.n_words = config.vocab_size
        self.pad_index = config.pad_index

        # model parameters 
        self.dim = config.emb_dim 
        self.hidden_dim = self.dim * 4 
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0 , "transformer dim must be a multiple of n_heads"

        # embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.dim)
        self.embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=config.layer_norm_eps)

        # transformer layers
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

    def forward(
            self,
            input_ids=None, # input_ids ~ [batch_size, max_len]
            attention_mask=None, # attention_mask ~ [batch_size, max_len]
            token_type_ids=None, # token_type_ids ~ None
            position_ids=None, # position_ids ~ None
            lengths=None, # lengths ~ None
            inputs_embeds=None, # inputs_embeds ~ None
            ):
 
        # input_ids ~ [batch_size, max_len] 
        # bs ~ batch_size  || slen ~ max_len
        bs, slen = input_ids.size()

        # device ~ cuda or cpu(depends on user)
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # lengths ~ [max_len - (count of pad tokens)]  || len(lengths) == batch_size
        lengths = (input_ids != self.pad_index).sum(dim=1).long()
            
        # check inputs
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen

        # mask ~ [batch_size, max_len]
        # attn_mask ~ [batch_size, max_len]
        mask, attn_mask = attention_mask, attention_mask 
        
        # position_ids
        position_ids = self.position_ids[:, :slen]

        # embeddings
        # inputs_embeds ~ [batch_size, max_len, emb_dim]
        inputs_embeds = self.embeddings(input_ids)

        # tensor ~ [batch_size, max_len, emb_size]
        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        # tensor ~ [batch_size, max_len, emb_size]
        tensor = self.layer_norm_emb(tensor)
        # tensor ~ [batch_size, max_len, emb_size]
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        # tensor ~ [batch_size, max_len, emb_size]
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        for i in range(self.n_layers):

            # self attention
            # attn_outputs ~ ([batch_size, max_len, emb_size])
            attn_outputs = self.attentions[i](
                    tensor, # tensor ~ [batch_size, max_len, emb_size]
                    attn_mask, # attn_mask ~ [batch_size, max_len]
                    )

            # attn ~ [batch_size, max_len, emb_size]
            attn = attn_outputs[0]
            # attn ~ [batch_size, max_len, emb_size]
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            # tensor ~ [batch_size, max_len, emb_size]
            tensor = tensor + attn
            # tensor ~ [batch_size, max_len, emb_size]
            tensor = self.layer_norm1[i](tensor)

            # FFN
            # tensor ~ [batch_size, max_len, emb_size]
            tensor = tensor + self.ffns[i](tensor)
            # tensor ~ [batch_size, max_len, emb_size]
            tensor = self.layer_norm2[i](tensor)
            # tensor ~ [batch_size, max_len, emb_size]
            tensor *= mask.unsqueeze(-1).to(dtype=tensor.dtype)

        return tuple(v for v in [tensor] if v is not None)

class XLMForSequenceClassification(XLMPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.num_labels = config.num_labels

        self.transformer = XLMModel(config)
        self.summary = nn.Linear(config.emb_dim, config.num_labels)
        self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.init_weights()

    def forward(
            self,
            input_ids=None, # input_ids ~ [batch_size, max_len]
            attention_mask=None, #attention_mask ~ [batch_size, max_len]
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            inputs_embeds=None,
            labels=None,
            ):

        # transformer_output ~ ([batch_size, max_len, emb_size])
        transformer_outputs = self.transformer(
                input_ids, # input_ids ~ [batch_size, max_len]
                attention_mask=attention_mask, # attention_mask ~ [batch_size, max_len]
                token_type_ids=token_type_ids, # token_type_ids ~ None
                position_ids=position_ids, # position_ids ~ None
                lengths=lengths, # lengths ~ None
                inputs_embeds=inputs_embeds, # inputs_embeds ~ None
                )

        # output ~ [batch_size, max_len, emb_size]
        output = transformer_outputs[0]
        # logits ~ [batch_size, emb_size]
        logits = output[:, 0]
        # logits ~ [batch_size, emb_size]
        logits = self.first_dropout(logits)
        # logits ~ [batch_size, num_labels]
        logits = self.summary(logits)

        loss = None
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)
        return ((loss,) + output) if loss is not None else output
