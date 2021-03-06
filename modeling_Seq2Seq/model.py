import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, p_drop=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.p_drop = p_drop
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.hidden_dim, self.num_layers, dropout=self.p_drop)
        self.dropout = nn.Dropout(self.p_drop)

    def forward(self, src): # src ~ [max_len, bs]
        embedded = self.embedding(src) # embedded ~ [max_len, bs, embedding_size]
        embedded = self.dropout(embedded) # embedded ~ [max_len, bs, embedding_size]
        outputs, (hidden, cell) = self.rnn(embedded) # hidden, cell ~ [num_layers, bs, hidden_size]
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, p_drop=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.p_drop = p_drop
        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.hidden_dim, self.num_layers, dropout=self.p_drop)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(self.p_drop)

    def forward(self, input, hidden, cell): # input ~ [bs] | hidden, cell ~ [num_layers, bs, hidden_size]
        input = input.unsqueeze(0) # input ~ [1, bs]
        embedded = self.embedding(input) # embedded ~ [1, bs, embedding_size]
        embedded = self.dropout(embedded) # embedded ~ [1, bs, embedding_size]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell)) # hidden, cell ~ [num_layers, bs, hidden_size] | output ~ [1, bs, hidden_size]
        prediction = self.fc_out(output.squeeze(0)) # prediction ~ [bs, trg_vocab_size]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5): # src ~ [max_len, bs] | trg ~ [max_len, bs]
        batch_size = trg.size(1) # batch_size ~ bs
        trg_len = trg.size(0) # trg_len ~ max_len
        trg_vocab_size = self.decoder.output_dim # trg_vocab_size
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device) # outputs ~ [max_len, bs, trg_vocab_size]
        hidden, cell = self.encoder(src) # hidden, cell ~ [num_layers, bs, hidden_size]
        input = trg[0, :] # First input to the decoder is <sos> token # input ~ [bs]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell) # output ~ [bs, trg_vocab_size] | hidden, cell ~ [num_layers, bs, hidden_size]
            outputs[t] = output # outputs ~ [max_len, bs, trg_vocab_size]
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs # outputs ~ [max_len, bs, trg_vocab_size]
