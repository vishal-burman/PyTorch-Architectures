############################
# Ignore this file
# Used for testing purposes
############################

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import OpenAIGPTTokenizer
from utils import Conv1D

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
# pad_token is not set by default
tokenizer.pad_token = '[PAD]'

class CustomDataset(Dataset):

    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.list_texts = []
        self.build()

    def __len__(self):
        return len(self.list_texts)

    def __getitem__(self, index):
        ids = self.list_texts[index]['input_ids']
        mask = self.list_texts[index]['attention_mask']
        
        return{
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long)
                }

    def build(self):
        for text in self.texts:
            self.list_texts.append(self.tokenizer(text, max_length=32, pad_to_max_length=True, truncation=True))

texts = ["this is my home", "that movie looks good", "this is a great book!", "what is your name?"]
dataset = CustomDataset(texts, tokenizer)
data_loader = DataLoader(dataset, shuffle=False, batch_size=2)
print("Length of DataLoader = ", len(data_loader))
emb_1 = nn.Embedding(40478, 768)
conv_1 = Conv1D(768, 768)
sample_1 = torch.ones(2, 32, 768)
sample_1 = conv_1(sample_1)
print(sample_1.shape)
#for sample in data_loader:
#    shape = sample['ids'].shape
#    ids = sample['ids']
#    x = emb_1(ids)
#    x = conv_1(x)
#    q, k, v = x.split(768, dim=2)
#    shape = x.size()[:-1] + (12, 768 // 12)
#    q = q.view(*shape)
#    print(q.shape)
#    break
