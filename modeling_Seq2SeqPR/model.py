import pdb
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, p_drop=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        self.rnn = nn.GRU(self.emb_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.p_drop)

    def forward(self, src): # src ~ [max_len, batch_size]
        embedded = self.dropout(self.embedding(src)) # embedded ~ [max_len, batch_size, embedding_size]
        outputs, hidden = self.rnn(embedded) # outputs, hidden ~ [batch_size, max_len, hidden_dim]
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, p_drop=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop
        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        self.rnn = nn.GRU(self.emb_dim + self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.emb_dim + self.hidden_dim * 2, self.output_dim)
        self.dropout = nn.Dropout(self.p_drop)

    def forward(self, input, hidden, context): # input ~ [batch_size] | hidden, context ~ [1, batch_size, hidden_size]
        input = input.unsqueeze(0) # input ~ [1, batch_size]
        embedded = self.dropout(self.embedding(input)) # embedded ~ [1, batch_size, embedding_size]
        emb_con = torch.cat((embedded, context), dim=2) # emb_con ~ [1, batch_size, embedding_size + hidden_size]
        output, hidden = self.rnn(emb_con, hidden) # output ~ [1, batch_size, hidden_dim] | hidden ~ [1, batch_size, hidden_dim]
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1) # output ~ [batch_size, embedding_size+ hidden_size + hidden_size]
        prediction = self.fc_out(output) # prediction ~ [batch_size, output_dim]
        return prediction, hidden


class Seq2SeqPR(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

