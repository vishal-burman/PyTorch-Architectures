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

texts = ['This is a good house', 'Where are you going?', 'There is someone at the door', 'What is your name?']
dataset = CustomDataset(texts, tokenizer)

data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=2)
for sample in data_loader:
    ids = sample['ids']
    mask = sample['mask']
    t_ids = sample['t_ids']
    break
###########################################

seq_length = ids.size()[1]

p_ids = torch.arange(512).expand((1, -1))
p_ids = p_ids[:, :seq_length]
em_1 = nn.Embedding(30522, 768)
em_2 = nn.Embedding(512, 768)
em_3 = nn.Embedding(30522, 768)

emb_1 = em_1(ids)
emb_2 = em_2(p_ids)
emb_3 = em_3(t_ids)
emb = emb_1 + emb_2 + emb_3

num_attention_heads = 12
attention_head_size = int(768/12)
print(attention_head_size)
all_head_size = 12 * attention_head_size

query = nn.Linear(768, all_head_size)
mixed_query = query(emb)
sample_x_size = mixed_query.size()[:-1] + (num_attention_heads, attention_head_size)
mixed_query = mixed_query.view(*sample_x_size).permute(0, 2, 1, 3)
print(mixed_query.shape, mixed_query.transpose(-1, 2).shape)
print(torch.matmul(mixed_query, mixed_query.transpose(-1, -2)).shape)
x_1 = torch.rand(2, 2, 2, 2)
x_2 = torch.rand(2, 2)
x_3 = nn.Softmax(dim=-2)(x_1@x_2)
print(x_1.contiguous().shape)
