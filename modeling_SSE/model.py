import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, att_unit, att_hops, n_layers):
        super().__init__()
        self.ut_dense = nn.Linear(hidden_dim * n_layers, att_unit, bias=False)
        self.et_dense = nn.Linear(att_unit, att_hops, bias=False)

    def forward(self, x): # x ~ [batch_size, seq_len, hidden_dim]
        ut = self.ut_dense(x) # ut ~ [batch_size, seq_len, att_unit]
        ut = torch.tanh(ut) # ut ~ [batch_size, seq_len, att_unit]
        et = self.et_dense(ut) # et ~ [batch_size, seq_len, att_hops]
        att = F.softmax(et.transpose(1, 2), dim=-1) # att ~ [batch_size, att_hops, seq_len]
        output = att @ x # output ~ [batch_size, att_hops, hidden_dim]
        return output, att

class BiLSTMSE(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, natt_unit, natt_hops, nfc, n_class, drop_prob, weights=None):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        if weights is not None:
            self.embedding_layer.weights = nn.Parameter(weights, requires_grad=False)
        self.bilstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=drop_prob, bidirectional=True)
        self.att_encoder = SelfAttention(hidden_dim, natt_unit, natt_hops, n_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.dense = nn.Linear(natt_hops * hidden_dim * n_layers, nfc)
        self.tanh = nn.Tanh()
        self.output_layer = nn.Linear(nfc, n_class)

    def forward(self, x):
        inp_emb = self.embedding_layer(x) # inp_emb ~ [batch_size, seq_len, emb_dim]
        h_output, (h_n, c_n) = self.bilstm(inp_emb.transpose(0, 1)) # h_output ~ [seq_len, batch_size, hidden_dim * n_layers]
        att_output, att = self.att_encoder(h_output.transpose(0, 1)) # att_output ~ [batch_size, att_hops, hidden_dim * n_layers]
        dense_input = self.dropout(torch.flatten(att_output, start_dim=1)) # dense_input ~ [batch_size, att_hops * hidden_dim * n_layers]
        dense_out = self.tanh(self.dense(dense_input)) # dense_out ~ [batch_size, nfc]
        output = self.output_layer(self.dropout(dense_out)) # output ~ [batch_size, n_class]
        return output, att
