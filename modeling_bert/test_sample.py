###############################################
# Ignore this file...used for testing of scripts
################################################
import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import BertForSequenceClassification
from transformers import BertTokenizer
from config_bert import BertConfig
config = BertConfig()
BertLayerNorm = nn.LayerNorm
#########################################
# Sample data code
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.train_list = []
        self.label_list = []
        self.build()

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        ids = self.train_list[index]['input_ids']
        mask = self.train_list[index]['attention_mask']
        target = self.label_list[index]

        return{
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(ids, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long)
                }

    def build(self):
        for t, l in zip(self.texts, self.labels):
            self.train_list.append(tokenizer(t, max_length=32, pad_to_max_length=True, truncation=True))
            self.label_list.append(l)
##########################################

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = ['This is a good house', 'That was a bad movie', 'This is amazing', 'That was horrible']
labels = [1, 0, 1, 0]
dataset = CustomDataset(texts, labels, tokenizer)

data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=2)

model = BertForSequenceClassification(config)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters = ", pytorch_total_params)
for sample in data_loader:
    output = model(input_ids = sample['ids'], attention_mask=sample['mask'], labels=sample['target'], return_dict=True)
    print("Loss = ", output[0])
    print("Logits shape = ", output[1].shape)
    break
