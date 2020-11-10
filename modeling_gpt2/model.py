import torch
import torch.nn as nn

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
