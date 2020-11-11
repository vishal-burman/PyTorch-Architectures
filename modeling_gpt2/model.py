import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        n_state = nx
        self.register_buffer("bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
        w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        outputs = [torch.matmul(w, v)]
        return outputs


class MLP(nn.Module):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)
    
    def forward(self, hidden_states, attention_mask=None):
        attn_outputs = self.attn(self.ln_1(hidden_states), attention_mask=attention_mask)
        attn_output = attn_outputs[1:]
        hidden_states = attn_output + hidden_states # residual connection
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states
        outputs = [hidden_states] + outputs
        return outputs

class GPT2Model(GPT2PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # TODO
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # TODO
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device).unsqueeze(0).view(-1, input_shape[-1])
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        for i, block in enumerate(self.h):
            outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        return tuple(v for v in [hidden_states] if v is not None)

class GPT2ForSequenceClassification(GPT2PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # TODO
        self.transformer = GPT2Model
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        # TODO
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        batch_size, sequence_length = input_ids.shape[:2]
        sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        pooled_logits = logits[range(batch_size), sequence_lengths]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        output = (pooled_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output
