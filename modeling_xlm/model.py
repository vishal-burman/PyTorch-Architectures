def get_masks(
        slen, # slen ~ max_len
        lengths,  # lengths ~ [max_len - count of pad tokens]  || len(lengths) = batch_size
        causal,  # False
        padding_mask=None, # padding_mask ~ [batch_size, max_len]
        ):
    """
    Generate hidden states mask, and optionally an attention mask
    """
    # alen ~ [1, max_len]
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    # padding_mask ~ [batch_size, max_len]
    if padding_mask is not None:
        # mask ~ [batch_size, max_len]
        mask = padding_mask
    else: # TODO check needed?
        assert lengths.max().item() <= slen
        mask = alen < lengths[:, None]
    
    # bs ~ batch_size
    bs = lengths.size(0)
    if causal: # TODO check needed?
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        # attn_mask ~ [batch_size, max_len]
        attn_mask = mask

    assert mask.size() == (bs, slen)
    return mask, attn_mask

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

    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        
        bs, qlen, dim = input.size()
        # TODO check needed
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """ projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        # TODO check_needed
        elif cache is None or self.layer_id is not in cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(k))

        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshape).expand_as(scores)
        scores.masked_fill_(mask, -float("inf"))

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        # TODO check needed
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)
        context = unshape(context)

        outputs = (self.out_lin(context),)
        # TODO check needed
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs




class XLMPretrainedModel(PretrainedModel):
    config_class = XLMConfig
    # TODO check if needed?
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

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

        # encoder / decoder, output layer
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError('XLM can be only used as an encoder')
        self.causal = config.causal

        # dictionary / languages
        self.n_langs = config.n_langs
        self.use_lang_emb = config.use_lang_emb
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index

        # model parameters TODO cross-check?
        self.dim = config.emb_dim 
        self.hidden_dim = self.dim * 4 
        self.n_heads = config.n_heads 
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0 , "transformer dim must be a multiple of n_heads"

        # embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.dim)
        if config.sinusoidal_embeddings:
            # TODO
            create_sinusoidal_embeddings(config.max_position_embeddings, self.dim, out=self.position_embeddings.weight)
        if config.n_langs > 1 and config.use_lang_emb:
            self.lang_embeddings = nn.Embedding(self.n_langs, self.dim)
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
            self.ffns.append(nn.TransformerFFN(self.dim, self.hidden_dim, self.dim, config=config))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))

        # TODO cross-check need for pruned heads?

        self.init_weights()
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand(1, -1))

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_input_embeddings):
        self.embeddings = new_embeddings

    # TODO cross-check need to implement prune heads logic

    def forward(
            self,
            input_ids=None, # input_ids ~ [batch_size, max_len]
            attention_mask=None, # attention_mask ~ [batch_size, max_len]
            langs=None, # langs ~ None
            token_type_ids=None, # token_type_ids ~ None
            position_ids=None, # position_ids ~ None
            lengths=None, # lengths ~ None
            cache=None, # cache ~ None
            head_mask=None, # head_mask ~ None
            inputs_embeds=None, # inputs_embeds ~ None
            output_attentions=None, # output_attentions ~ None
            output_hidden_states=None, # output_hidden_states ~ None
            return_dict=None,
            ):

        # output_attentions ~ False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states ~ False
        output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
        # TODO check needed?
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input_ids ~ [batch_size, max_len]
        if input_ids is not None:
            # bs ~ batch_size  || slen ~ max_len
            bs, slen = input_ids.size()
        else: # TODO check needed?
            bs, slen = inputs_embeds.size()[:-1]

        # device ~ cuda or cpu(depends on user)
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if lengths is None:
            if input_ids is not None:
                # lengths ~ [max_len - (count of pad tokens)]  || len(lengths) == batch_size
                lengths = (input_ids != self.pad_index).sum(dim=1).long()
            else: # TODO check needed?
                lengths = torch.tensor([slen] * bs, device=device)

        # check inputs
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen

        # generate masks TODO
        # mask ~ [batch_size, max_len]
        # attn_mask ~ [batch_size, max_len]
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)

        # do not recompute cached elements TODO needed ?
        if cache is not None and input_ids is not None:
            _slen = slen - cache['slen']
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        # input_embeds ~ None
        if inputs_embeds is None:
            # inputs_embeds ~ [batch_size, max_len, emb_dim]
            inputs_embeds = self.embedding(input_ids)

        # tensor ~ [batch_size, max_len, emb_size]
        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        # TODO check needed? --> doesn't get accessed
        if langs is not None and self.use_lang_emb and self.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)
        # TODO check needed? --> doesn't get accessed
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        # tensor ~ [batch_size, max_len, emb_size]
        tensor = self.layer_norm(tensor)
        # tensor ~ [batch_size, max_len, emb_size]
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        # tensor ~ [batch_size, max_len, emb_size]
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        # hidden_states ~ None TODO check --> not accessed
        hidden_states = () if output_hidden_states else None
        # TODO needed? check --> not accessed
        attentions = () if output_attentions else None
        for i in range(self.n_layers):
            # TODO check if needed?
            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            attn_outputs = self.attention[i](
                    tensor, # tensor ~ [batch_size, max_len, emb_size]
                    attn_mask, # attn_mask ~ [batch_size, max_len]
                    cache=cache, # cache ~ None --> check needed TODO
                    head_mask=head_mask[i], # head_mask ~ None --> check needed TODO
                    output_attentions=output_attentions, # output_attentions ~ None --> check needed TODO
                    )
            attn = attn_outputs[0]
            # TODO check if needed?
            if output_attentions:
                attentions = attentions + (attn_outputs[1:],)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(dtype=tensor.dtype)

        # Add last hidden state TODO check if needed?
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # Update cache length TODO check if needed?
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # TODO check if needed?
        if not return_dict:
            return tuple(v for v in [tensor, hidden_states, attentions] if v is not None)




class XLMForSequenceClassification(XLMPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # TODO
        self.transformer = XLMModel(config)
        # TODO needed?
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None, # input_ids ~ [batch_size, max_len]
            attention_mask=None, #attention_mask ~ [batch_size, max_len]
            langs=None, # langs ~ optional(None)
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            cache=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
                input_ids, # input_ids ~ [batch_size, max_len]
                attention_mask=attention_mask, # attention_mask ~ [batch_size, max_len]
                langs=langs, # langs ~ None
                token_type_ids=token_type_ids, # token_type_ids ~ None
                position_ids=position_ids, # position_ids ~ None
                lengths=lengths, # lengths ~ None
                cache=cache, # cache ~ None
                head_mask=head_mask, # head_mask ~ None
                inputs_embeds=inputs_embeds, # inputs_embeds ~ None
                output_attentions=output_attentions, # output_attentions ~ None
                output_hidden_states=output_hidden_states, # output_hidden_states ~ None
                return_dict=return_dict,
                )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

