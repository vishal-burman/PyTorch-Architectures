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
                'target': torch.tensor(target, dtype=torch.long).unsqueeze(0)
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

texts_train = texts[:1000]
labels_train = labels[:1000]

texts_valid = texts[9500:]
labels_valid = labels[9500:]

start_time = time.time()
train_dataset = CustomDataset(texts_train, labels_train, tokenizer)
valid_dataset = CustomDataset(texts_valid, labels_valid, tokenizer)
print("Dataset Conversion Done!!")
print("Time Taken = ", (time.time() - start_time)/60)

BATCH_SIZE = 2
LEARNING_RATE = 1e-05
EPOCHS = 5

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=BATCH_SIZE)
print("Total train batches = ", len(train_loader))
print("Total valid batches = ", len(valid_loader))

model = BertForSequenceClassification(config).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters = ", pytorch_total_params)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    model.eval()
    for idx, sample in enumerate(data_loader):
        ids = sample['ids'].to(device)
        mask = sample['mask'].to(device)
        target = sample['target'].to(device)

        output = model(input_ids=ids, attention_mask=mask)
        logits = output[0]
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += target.size(0)
        correct_pred += (predicted_labels.unsqueeze(1) == target).sum()
    return correct_pred.float()/num_examples*100
        

start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    for idx, sample in enumerate(train_loader):
        ids = sample['ids'].to(device)
        mask = sample['mask'].to(device)
        target = sample['target'].to(device)
        
        optimizer.zero_grad()
        output = model(input_ids=ids, attention_mask=mask, labels=target, return_dict=True)
        loss = output[0]

        if idx % 100 == 0:
            print('Loss = ', loss.item())
        
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.set_grad_enabled(False):

        train_acc = compute_accuracy(model, train_loader, device)
        valid_acc = compute_accuracy(model, valid_loader, device)

        print("Train Accuracy = ", train_acc)
        print("Valid Accuracy = ", valid_acc)

    elapsed_time = (time.time() - start_time) / 60
    print("Elapsed Time: ", elapsed_time)
