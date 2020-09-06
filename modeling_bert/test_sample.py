###############################################
# Ignore this file...used for testing of scripts
################################################
import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import BertEmbeddings
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

embeddings = BertEmbeddings(config)
for sample in data_loader:
    embeddings = embeddings(input_ids=sample['ids'])
    break
