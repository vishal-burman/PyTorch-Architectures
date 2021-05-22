import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.p_drop)
        self.dense_1 = nn.Linear(config.dim, config.expanded_dim)
        self.dense_2 = nn.Linear(config.expanded_dim, config.dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class FourierLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft2(x).real
        return x

class PreNormResidual(nn.Module):
    def __init__(self, config, fn):
        super().__init__()
        self.fn = fn
        self.layer_norm = nn.LayerNorm(config.dim, eps=config.eps)

    def forward(self, x):
        x = self.fn(self.layer_norm(x)) + x
        return x

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embed, config.dim)
        self.layer_norm = nn.LayerNorm(config.dim, eps=config.eps)
        self.dropout = nn.Dropout(config.p_drop)
    
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.word_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class FNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([])
        for _ in range(config.depth):
            self.layers.append(nn.ModuleList([
                PreNormResidual(config, FourierLayer()),
                PreNormResidual(config, FeedForwardLayer(config)),
                ]))

    def forward(self, input_ids):
        embeds = self.embeddings(input_ids)
        for fft, ff in self.layers:
            embeds = fft(embeds) 
            embeds = ff(embeds)
        return embeds

class FNetClassify(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fnet = FNetModel(config)
        self.num_labels = config.num_labels
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.p_drop)
        self.relu = nn.ReLU()

    def forward(self, input_ids, labels):
        fnet_output = self.fnet(input_ids)
        pooled_output = fnet_output[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits)
