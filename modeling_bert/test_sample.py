###############################################
# Ignore this file...used for testing of scripts
################################################
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


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
print("Tokenizer Loaded")

texts = ['This is a good house', 'Where are you going?', 'There is someone at the door', 'What is your name?']
dataset = CustomDataset(texts, tokenizer)
print("Dataset Done")

data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=2)
for sample in data_loader:
    ids = sample['ids']
    mask = sample['mask']
    t_ids = sample['t_ids']
    break
###########################################

print(ids.shape)
seq_length = ids.size()[1]
print(seq_length)

p_ids = torch.arange(512).expand((1, -1))
p_ids = p_ids[:, :seq_length]
print(ids.shape, p_ids.shape)
em_1 = nn.Embedding(30522, 768)
em_2 = nn.Embedding(512, 768)
em_3 = nn.Embedding(30522, 768)

emb_1 = em_1(ids)
emb_2 = em_2(p_ids)
emb_3 = em_3(t_ids)
print(emb_1.shape, " ", emb_2.shape, " ", emb_3.shape)
emb = emb_1 + emb_2 + emb_3
print(emb.shape)
