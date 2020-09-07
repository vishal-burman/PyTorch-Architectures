###############################################
# Main file 
################################################
import time
import csv
import sys
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import BertForSequenceClassification
from transformers import BertTokenizer
from config_bert import BertConfig
config = BertConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

texts = []
labels = []
count = 0
with open("dataset.csv", "r") as file_1:
    reader = csv.reader(file_1)
    for line in reader:
        if count == 10001:
            break
        texts.append(line[0].strip())
        labels.append(line[1].strip())
        count += 1

texts = texts[1:]
labels = labels[1:]

labels = [1 if label == "positive" else 0 for label in labels]

texts_train = texts[:9000]
labels_train = labels[:9000]

texts_valid = texts[9000:]
labels_valid = labels[9000:]

start_time = time.time()
train_dataset = CustomDataset(texts_train, labels_train, tokenizer)
valid_dataset = CustomDataset(texts_valid, labels_valid, tokenizer)
print("Dataset Conversion Done!!")
print("Time Taken = ", (time.time() - start_time)/60)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=2)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=2)

model = BertForSequenceClassification(config).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters = ", pytorch_total_params)


for sample in train_loader:
    ids = sample['ids'].to(device)
    mask = sample['mask'].to(device)
    target = sample['target'].to(device)
    output = model(input_ids=ids, attention_mask=mask, labels=target, return_dict=True)
    print("Loss = ", output[0])
    print("Logits shape = ", output[1].shape)
    break
