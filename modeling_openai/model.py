import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        # n_state ~ 768 where nx = n_embed = 768
        n_state = nx

        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        # self.n_head ~ 12
        self.n_head = config.n_head
        # self.split_size ~ 768
        self.split_size = n_state
        # scale ~ False
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.bias[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1e4 * (1-b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, 
            x, # x ~ [batch_size, max_len, emb_size]
            attention_mask=None, # attention_mask ~ [batch_size, 1, 1, max_len]
            head_mask=None, # head_mask ~ None
            output_attentions=False):

        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a] + attn_outputs[1:]
        return outputs


class MLP(nn.Module):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embed
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4*nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, 
            x, # x ~ [batch_size, max_len, emb_size]
            attention_mask=None, # attention_mask ~ [batch_size, 1, 1, max_len]
            head_mask=None, # head_mask ~ None 
            output_attentions=False):

        attn_outputs = self.attn(
                x,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                )
        a = attn_outputs[0]

        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = [h] + attn_outputs[1:]
        return outputs

class OpenAIGPTPretrainedModel(PretrainedModel):
    config_class = OpenAIGPTConfig
    base_model_prefix = "transformer"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        Prepare the head_mask if needed
        """
        head_mask = [None] * num_layers
        return head_mask

class OpenAIGPTModel(OpenAIGPTPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])

        self.register_buffer("position_ids", torch.arange(config.n_positions))
        self.init_weights()

    def get_input_embeddings(self):
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        self.tokens_embed = new_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
            self,
            input_ids=None, # input_ids ~ [batch_size, max_len]
            attention_mask=None, # attention_mask ~ [batch_size, max_len]
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None, # output_attentions ~ False
            output_hidden_states=None, # output_hidden_states ~ False
            return_dict=None,
            ):

        # output_attentions ~ False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states ~ False
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input_ids ~ [batch_size, max_len] || inputs_embeds ~ None
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Both input_ids and inputs_embeds is specified!")
        # input_ids ~ [batch_size, max_len]
        elif input_ids is not None:
            # input_shape ~ [batch_size, max_len]
            input_shape = input_ids.shape()
            # input_ids ~ [batch_size, max_len]
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Specify either input_ids or inputs_embeds")

        # position_ids ~ [n_positions] where n_positions = 512
        if position_ids is None:
            # position_ids ~ [1, max_len]
            position_ids = self.position_ids[None, :input_shape[-1]]


        # attention_mask ~ [batch_size, max_len]
        if attention_mask is not None:
            # attention_mask ~ [batch_size, 1, 1, max_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # TODO check? On single GPU
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility

            # Invert attention masks
            # attention_mask ~ [batch_size, 1, 1, max_len]
            attention_mask = (1.0 - attention_mask) * -10000.0

        # head_mask ~ [None] * num_hidden_layers
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # inputs_embeds ~ None 
        if inputs_embeds is None:
            # inputs_embeds ~ [batch_size, max_len, emb_size] where emb_size = 512
            inputs_embeds = self.tokens_embed(input_ids)
        # position_embeds ~ [1, max_len, emb_size]
        position_embeds = self.positions_embed(position_ids)
        # token_ids ~ None
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        # hidden_states ~ [batch_size, max_len, emb_size]
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        # hidden_states ~ [batch_size, max_len, emb_size]
        hidden_states = self.drop(hidden_states)

        # output_shape ~ torch.Size([batch_size, max_len, emb_size])
        output_shape = input_shape + (hidden_states.size(-1),)

        # all_attentions ~ None
        all_attentions = () if output_attentions else None
        # all_hidden_states ~ None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            # TODO needed?
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
                
            outputs = block(hidden_states, attention_mask, head_mask[i], output_attentions=output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = hidden_states.view(*output_shape)
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class OpenAIGPTLMHeadModel(OpenAIGPTPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids=None, # input_ids ~ [batch_size, max_len]
            attention_mask=None, # attention_mask ~ [batch_size, max_len]
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            input_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):

        # TODO
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens 
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
