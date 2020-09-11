
class OpenAIGPTModel(OpenAIGPTPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # TODO
        pass
    def get_input_embeddings(self):
        # TODO
        pass
    def set_input_embeddings(self):
        # TODO
        pass
    def _prune_heads(self, heads_to_prune):
        # TODO
        pass

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            input_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):
        # TODO
        pass

class OpenAIGPTLMHeadModel(OpenAIGPTPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        # TODO
        pass

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
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

        pass
