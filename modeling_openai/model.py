import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from utils import Conv1D, gelu_new
from config_openai import OpenAIGPTConfig

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        n_state = nx # n_state ~ 768 where nx = n_embed = 768
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head # self.n_head ~ 12
        self.split_size = n_state # self.split_size ~ 768
        self.scale = scale # scale ~ True
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def _attn(self, q, k, v, attention_mask=None) # q, v ~ [batch_size, num_heads, max_len, emb_size // num_heads] k ~ [batch_size, num_heads, emb_size //num_heads, max_len]
        w = torch.matmul(q, k) # w ~ [batch_size, 12, max_len, max_len]
        w = w / math.sqrt(v.size(-1)) # w ~ [batch_size, 12, max_len, max_len]
        b = self.bias[:, :, :w.size(-2), :w.size(-1)] # b ~ [batch_size, 12, max_len, max_len]
        w = w * b + -1e4 * (1-b) # w ~ [batch_size, 12, max_len, max_len]  ~ unidirectional attention
        w = w + attention_mask # w ~ [batch_size, 12, max_len, max_len]
        w = nn.Softmax(dim=-1)(w) # w ~ [batch_size, 12, max_len, max_len]
        w = self.attn_dropout(w) # w ~ [batch_size, 12, max_len, max_len] 
        outputs = [torch.matmul(w, v)] # outputs ~ [[batch_size, 12, max_len, 64]]
        return outputs

    def forward(self, x, attention_mask=None) # x ~ [batch_size, max_len, emb_size] && attention_mask ~ [batch_size, 1, 1, max_len]
        bs, slen = x.shape[:2] # bs ~ batch_size, slen ~ max_len
        x = self.c_attn(x) # x ~ [batch_size, max_len, n_state * 3] where (n_state * 3 = 2034)
        query, key, value = x.split(self.split_size, dim=2) # query, key, value ~ [batch_size, max_len, emb_size]
        query = query.view(bs, slen, self.n_head, -1).permute(0, 2, 1, 3) # query ~ [batch_size, 12, max_len, 64] 
        key = key.view(bs, slen, self.n_head, -1).permute(0, 2, 3, 1) # key ~ [batch_size, 12, 64, max_len]
        value = value.view(bs, slen, self.n_head, -1).permute(0, 2, 1, 3) # value ~ [batch_size, 12, max_len, 64]
        attn_outputs = self._attn(query, key, value, attention_mask) # attn_outputs ~ [[batch_size, 12, max_len, 64]]
        a = attn_outputs[0] # a ~ [batch_size, 12, max_len, 64]
        a = a.permute(0, 2, 1, 3).contiguous().view(bs, slen, -1) # a ~ [batch_size, max_len, 768]
        a = self.c_proj(a) # a ~ [batch_size, max_len, 768]
        a = self.resid_dropout(a) # a ~ [batch_size, max_len, 768]
        outputs = [a] + attn_outputs[1:] # outputs ~ [[batch_size, max_len, 768]]
        return outputs

class MLP(nn.Module):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x): # x ~ [batch_size, max_len, 768]
        h = self.act(self.c_fc(x)) # h ~ [batch_size, max_len, 768]
        h2 = self.c_proj(h) # h2 ~ [batch_size, max_len, 768]
        return self.dropout(h2)

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False): # scale ~ True
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4*nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None):
        attn_outputs = self.attn(x, attention_mask=attention_mask) # attn_outputs ~ [[batch_size, max_len, 768]] where emb_size = 768
        a = attn_outputs[0] # a ~ [batch_size, max_len, 768]
        n = self.ln_1(x + a) # n ~ [batch_size, max_len, 768]
        m = self.mlp(n) # m ~ [batch_size, max_len, 768]
        h = self.ln_2(n + m) # h ~ [batch_size, max_len, 768]
        outputs = [h] + attn_outputs[1:] # outputs ~ [[batch_size, max_len, 768]]
        return outputs

class OpenAIGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.register_buffer("position_ids", torch.arange(config.n_positions)) 

    def forward(self, input_ids=None, attention_mask=None): # input_ids, attention_mask ~ [batch_size, max_len]
        input_shape = input_ids.shape # input_shape ~ [batch_size, max_len]
        input_ids = input_ids.view(-1, input_shape[-1]) # input_ids ~ [batch_size, max_len]
        position_ids = self.position_ids[None, :input_shape[-1]] # position_ids ~ [1, max_len]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # attention_mask ~ [batch_size, 1, 1, max_len]
        attention_mask = (1.0 - attention_mask) * -10000.0 # attention_mask ~ [batch_size, 1, 1, max_len]
        inputs_embeds = self.tokens_embed(input_ids) # inputs_embeds ~ [batch_size, max_len, emb_size] where emb_size = 512
        position_embeds = self.positions_embed(position_ids) # position_embeds ~ [1, max_len, emb_size]
        hidden_states = inputs_embeds + position_embeds # hidden_states ~ [batch_size, max_len, emb_size]
        hidden_states = self.drop(hidden_states) # hidden_states ~ [batch_size, max_len, emb_size]
        output_shape = input_shape + (hidden_states.size(-1),) # output_shape ~ torch.Size([batch_size, max_len, emb_size])
        for i, block in enumerate(self.h):
            outputs = block(hidden_states, attention_mask) # outputs ~ [[batch_size, max_len, 768]]
            hidden_states = outputs[0] # hidden_states ~ [batch_size, max_len, 768]        
        hidden_states = hidden_states.view(*output_shape) # hidden_states ~ [batch_size, max_len, 768]
        return tuple(v for v in [hidden_states] if v is not None) # return ~ ([batch_size, max_len, emb_size])

class OpenAIGPTLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None): 
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask) # transformer_outputs ~ ([batch_size, max_len, emb_size])
        hidden_states = transformer_outputs[0] # hidden_states ~ [batch_size, max_len, 768]
        lm_logits = self.lm_head(hidden_states) # hidden_states ~ [batch_size, max_len, vocab_size] where vocab_size = 40478
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous() # shift_logits ~ [batch_size, max_len -1, vocab_size]
            shift_labels = labels[..., 1:].contiguous() # shift_labels ~ [batch_size, max_len - 1]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        output = (lm_logits,) + transformer_outputs[1:] # output ~ ([batch_size, max_len, vocab_size])
        return ((loss,) + output) if loss is not None else output # return ~ (loss, ([batch_size, max_len, vocab_size]))
