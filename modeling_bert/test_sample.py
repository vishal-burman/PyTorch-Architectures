###############################################
# Ignore this file...used for testing of scripts
################################################
import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config_bert import BertConfig

config = BertConfig()
BertLayerNorm = nn.LayerNorm
#########################################
# Sample data code
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
        self.train_list = []
        self.build()

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        ids = self.train_list[index]['input_ids']
        mask = self.train_list[index]['attention_mask']
        t_ids = self.train_list[index]['token_type_ids']

        return{
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(ids, dtype=torch.long),
                't_ids': torch.tensor(ids, dtype=torch.long)
                }

    def build(self):
        for t in self.texts:
            self.train_list.append(tokenizer(t, max_length=32, pad_to_max_length=True, truncation=True, return_token_type_ids=True))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = ['This is a good house', 'Where are you going?', 'There is someone at the door', 'What is your name?']
dataset = CustomDataset(texts, tokenizer)

data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=2)

for sample in data_loader:
    sample_test = sample
    ids = sample['ids']
    mask = sample['mask']
    t_ids = sample['t_ids']
    break
###########################################

class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Seems a bit hacky (alternative??) 
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            # input_ids ~ [batch_size, seq_max_len]
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # position_ids ~ [1, seq_max_len]
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            # token_type_ids ~ [batch_size, seq_len] 
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # inputs_embeds ~ [batch_size, seq_max_len, emb_size]
            inputs_embeds = self.word_embeddings(input_ids)
        # position_embeds ~ [1, max_seq_len, emb_size]
        position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeds ~ [batch_size, max_seq_len, emb_size]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings ~ [batch_size, max_seq_len, emb_size]
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # embeddings ~ [batch_size, max_seq_len, emb_size]
        embeddings = self.LayerNorm(embeddings)
        # embeddings ~ [batch_size, max_seq_len, emb_size]
        embeddings = self.dropout(embeddings)
        return embeddings
###############################################################

embeddings = BertEmbeddings(config)
for sample in data_loader:
    embeddings = embeddings(input_ids=sample['ids'])
    break
