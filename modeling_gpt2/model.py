import torch
import torch.nn as nn

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
