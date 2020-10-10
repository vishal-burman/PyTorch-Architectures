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
        self.dim = config.emb_dim # 512 by default
        self.hidden_dim = config.hidden_dim # 2048 by default
        self.n_heads = config.n_heads # 8 by default
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
            input_ids=None,
            attention_mask=None,
            langs=None,
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            cache=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            bs, slen = input_ids.size()
        else:
            bs, slen = inputs_embeds.size()[:-1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if lengths is None:
            if input_ids is not None:
                lengths = (input_ids != self.pad_index).sum(dim=1).long()
            else:
                lengths = torch.tensor([slen] * bs, device=device)

        # check inputs
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen

        # generate masks TODO
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
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        if langs is not None and self.use_lang_emb and self.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        hidden_states = () if output_hidden_states else None
        # TODO needed?
        attentions = () if output_attentions else None
        for i in range(self.n_layers):
            # TODO check if needed?
            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            attn_outputs = self.attention[i](
                    tensor,
                    attn_mask,
                    cache=cache,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
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
            input_ids=None,
            attention_mask=None,
            langs=None,
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
                input_ids,
                attention_mask=attention_mask,
                langs=langs,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                lengths=lengths,
                cache=cache,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
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

