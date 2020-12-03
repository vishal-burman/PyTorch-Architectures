import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights=None):
        super().__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.weights = weights

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if self.weights is not None:
            self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        self.dropout = 0.8
        self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True, num_layers=2)
        self.W_s1 = nn.Linear(2*hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30*2*hidden_size, 2000)
        self.label = nn.Linear(2000, output_size)
        self.register_buffer("h_0", torch.zeros(4, self.batch_size, self.hidden_size))
        self.register_buffer("c_0", torch.zeros(4, self.batch_size, self.hidden_size))

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def forward(self, x):
        input = self.word_embeddings(x)
        input = input.permute(1, 0, 2)
        output, (h_n, c_n) = self.bilstm(input, (self.h_0, self.c_0))
        output = output.permute(1, 0, 2)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = attn_weight_matrix @ output
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size(1) * hidden_matrix.size(2)))
        logits = self.label(fc_out)
        return logits
