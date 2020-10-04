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

        # 


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

