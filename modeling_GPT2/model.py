import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv1D, gelu_new

class Attention(nn.Module):
    def __init__(self, nx, n_ctx):
        super().__init__()
        n_state = nx
        self.register_buffer("bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = 12
        self.split_size = n_state
        self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def _attn(self, q, k, v, attention_mask=None): # q, v ~ [batch_size, n_head, seq_len, emb_size // n_head] || k ~ [batch_size, n_head, emb_size // n_head, seq_len]
        w = torch.matmul(q, k) # w ~ [batch_size, n_head, seq_len, seq_len]
        w = w / (float(v.size(-1)) ** 0.5) # w ~ [batch_size, n_head, seq_len, seq_len]
        nd, ns = w.size(-2), w.size(-1) # TODO needed?
        mask = self.bias[:, :, ns - nd : ns, :ns] # mask ~ [1, 1, seq_len, seq_len]
        w = w * mask + self.masked_bias * (1 - mask) # w ~ [batch_size, n_head, seq_len, seq_len]
        w = w + attention_mask # w ~ [batch_size, n_head, seq_len, seq_len]
        w = self.attn_dropout(F.softmax(w, dim=-1)) # w ~ [batch_size, n_head, seq_len, seq_len]
        outputs = [torch.matmul(w, v)] # outputs [[batch_size, n_head, seq_len, emb_size // n_head]]
        return outputs  

    def forward(self, hidden_states, attention_mask=None): # hidden_states ~ [batch_size, seq_len, emb_dim] || attention_mask ~ [batch_size, 1, 1, seq_len]
        bs, seq_len = hidden_states.shape[:2]
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2) # query, key, value ~ [batch_size, seq_len, emb_dim]
        query = query.reshape(bs, self.n_head, seq_len, -1) # query ~ [batch_size, n_head, seq_len, emb_dim // n_head]
        key = key.reshape(bs, self.n_head, seq_len, -1).transpose(-1, -2) # key ~ [batch_size, n_head, emb_dim // n_head, seq_len]
        value = value.reshape(bs, self.n_head, seq_len, -1) # value ~ [batch_size, n_head, seq_len, emb_dim // n_head]
        attn_outputs = self._attn(query, key, value, attention_mask) # attn_outputs ~ [[batch_size, n_head, seq_len, emb_size // n_head]]
        a = attn_outputs[0] # a ~ [batch_size, n_head, seq_len, emb_size // n_head]
        a = a.permute(0, 2, 1, 3).reshape(bs, seq_len, -1) # a ~ [batch_size, seq_len, emb_size]
        a = self.c_proj(a) # a ~ [batch_size, seq_len, emb_size]
        a = self.resid_dropout(a) # a ~ [batch_size, seq_len, emb_size]
        outputs = [a] # outputs ~ [[batch_size, seq_len, emb_size]]
        return outputs

class MLP(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        nx = 768
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu_new
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x): # x ~ [batch_size, seq_len, emb_size]
        h = self.act(self.c_fc(x)) # h ~ [batch_size, seq_len, 4 * emb_size]
        h2 = self.c_proj(h) # h2 ~ [batch_size, seq_len, emb_size]
        return self.dropout(h2)

class Block(nn.Module):
    def __init__(self, n_ctx):
        super().__init__()
        hidden_size = 768
        inner_dim = 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attn = Attention(hidden_size, n_ctx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = MLP(inner_dim)
    
    def forward(self, hidden_states, attention_mask=None): # hidden_states ~ [batch_size, seq_len, emb_dim] || attention_mask ~ [batch_size, 1, 1, seq_len]
        attn_outputs = self.attn(self.ln_1(hidden_states), attention_mask=attention_mask) # attn_outputs ~ [[batch_size, seq_len, emb_dim]]
        attn_output = attn_outputs[0] # attn_output ~ [batch_size, seq_len, emb_dim]
        hidden_states = attn_output + hidden_states # hidden_states ~ [batch_size, seq_len, emb_dim] --> residual connection
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states)) # feed_forward_hidden_states ~ [batch_size, seq_len, emb_size]
        hidden_states = hidden_states + feed_forward_hidden_states # hidden_states ~ [batch_size, seq_len, emb_size] --> residual connection
        outputs = [hidden_states] # outputs [[batch_size, seq_len, emb_size]]
        return outputs

class GPT2Classify(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50257, 768)
        self.wpe = nn.Embedding(1024, 768)
        self.drop = nn.Dropout(0.1)
        self.h = nn.ModuleList([Block(1024) for _ in range(3)])
        self.ln_f = nn.LayerNorm(768, eps=1e-5)
        self.score = nn.Linear(768, 2, bias=False)

    def forward(self, input_ids, attention_mask, labels=None): # input_ids ~ [batch_size, seq_len] || attention_mask ~ [batch_size, seq_len]
        input_shape = input_ids.size() # input_shape ~ [batch_size, seq_len]
        input_ids = input_ids.view(-1, input_shape[-1]) # input_ids ~ [batch_size, seq_len] TODO needed?
        batch_size = input_ids.shape[0] 
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device).unsqueeze(0).view(-1, input_shape[-1]) # position_ids ~ [1, seq_len]
        attention_mask = attention_mask[:, None, None, :] # attention_mask ~ [batch_size, 1, 1, seq_len]
        attention_mask = (1.0 - attention_mask) * -10000.0 # attention_mask ~ [batch_size, 1, 1, seq_len]
        inputs_embeds = self.wte(input_ids) # inputs_embeds ~ [batch_size, seq_len, emb_dim]
        position_embeds = self.wpe(position_ids) # position_embeds ~ [batch_size, seq_len, emb_dim]
        hidden_states = inputs_embeds + position_embeds # hidden_states ~ [batch_size, seq_len, emb_dim]
        hidden_states = self.drop(hidden_states) # hidden_states ~ [batch_size, seq_len, emb_dim]
        output_shape = input_shape + (hidden_states.size(-1),)
        for i, block in enumerate(self.h):
            outputs = block(hidden_states, attention_mask=attention_mask) # outputs ~ [[batch_size, seq_len, emb_size]]
            hidden_states = outputs[0] # hidden_states ~ [batch_size, seq_len, emb_size]
        hidden_states = self.ln_f(hidden_states) # hidden_states ~ [batch_size, seq_len, emb_size]
        hidden_states = hidden_states.view(*output_shape) # hidden_states ~ [batch_size, seq_len, emb_size] TODO needed?
        logits = self.score(hidden_states) # logits ~ [batch_size, seq_len, num_classes]
        pooled_logits = logits[range(input_ids.size(0)), -1] # pooled_logits ~ [batch_size, num_classes]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, 2), labels.view(-1))
        output = (pooled_logits,) # output ~ ([batch_size, num_classes])
        return ((loss,) + output) if loss is not None else output # return ~ (loss, [batch_size, num_labels])
