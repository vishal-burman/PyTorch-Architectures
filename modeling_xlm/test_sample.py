########################################
# This file is used to debug module
# IGNORE THIS
########################################

import sys
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import XLMTokenizer
from model import XLMForSequenceClassification
from config_xlm import XLMConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = XLMConfig()
config.n_layers = 1
tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
model = XLMForSequenceClassification(config).to(device)

########################################################################
# Dataset
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
                'target': torch.tensor(target, dtype=torch.long).unsqueeze(0),
                }

    def build(self):
        for t, l in zip(self.texts, self.labels):
            self.train_list.append(tokenizer(t, max_length=128, pad_to_max_length=True, truncation=True))
            self.label_list.append(l)
#########################################################################

#texts = []
#labels = []
#with open("dataset.csv", "r") as file_1:
#    reader = csv.reader(file_1)
#    for line in reader:
#        texts.append(line[0].strip())
#        labels.append(line[1].strip())
#
#texts = texts[1:]
#labels = labels[1:]
#
#labels = [1 if label == "positive" else 0 for label in labels]
#
#texts_train = texts[:10000]
#labels_train = labels[:10000]
#
#texts_valid = texts[10000:10100]
#labels_valid = labels[10000:10100]
#
#start_time = time.time()
#train_dataset = CustomDataset(texts_train, labels_train, tokenizer)
#valid_dataset = CustomDataset(texts_valid, labels_valid, tokenizer)
#print("Dataset Conversion Done!!")
#print("Time Taken = ", (time.time() - start_time)/60)
#torch.save(train_dataset, 'train_dataset.pt')
#torch.save(valid_dataset, 'valid_dataset.pt')
#sys.exit("Break")
#########################################################################
train_dataset = torch.load('train_dataset.pt')
valid_dataset = torch.load('valid_dataset.pt')

BATCH_SIZE = 4
LR = 5e-04
EPOCHS = 10

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=BATCH_SIZE)
print("Length of Train DataLoader: ", len(train_loader))
print("Length of Valid DataLoader: ", len(valid_loader))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

############################################################################
def compute_accuracy(model, data_loader, device):
  correct_pred, num_examples = 0, 0
  model.eval()
  with torch.set_grad_enabled(False):
    for sample in data_loader:
      ids = sample['ids'].to(device)
      mask = sample['mask'].to(device)
      labels = sample['target'].to(device)

      outputs = model(input_ids=ids, attention_mask=mask)
      logits = outputs[0]
      probas = F.softmax(logits, dim=1)
      _, predicted_labels = torch.max(probas, 1)
      num_examples += labels.size(0)
      correct_pred += (predicted_labels.unsqueeze(1) == labels).sum()
    return correct_pred.float() / num_examples * 100

def compute_loss(model, data_loader, device):
  total_loss = 0
  model.eval()
  with torch.set_grad_enabled(False):
    for sample in data_loader:
      ids = sample['ids'].to(device)
      mask = sample['mask'].to(device)
      labels = sample['target'].to(device)

      outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
      loss = outputs[0]
      total_loss += loss.item()
  return (total_loss / len(data_loader))

start_time = time.time()
for epoch in range(EPOCHS):
  model.train()
  for idx, sample in enumerate(train_loader):
    ids = sample['ids'].to(device)
    mask = sample['mask'].to(device)
    labels = sample['target'].to(device)
    
    optimizer.zero_grad()
    
    logits = model(input_ids=ids, attention_mask=mask, labels=labels)
    loss = logits[0]

    # LOGGING
    if idx % 100 == 0:
      print("Batch: %04d/%04d || Epoch: %03d/%03d" % (idx, len(train_loader), epoch+1, EPOCHS))

    loss.backward()
    optimizer.step()

  model.eval()
  with torch.set_grad_enabled(False):
    train_loss = compute_loss(model, train_loader, device)
    # valid_loss = compute_loss(model, valid_loader, device)
    valid_accuracy = compute_accuracy(model, valid_loader, device)
    # print("Train Loss: ", train_loss)
    print("Train Loss: %.3f" % (train_loss))
    print("Valid Accuracy: ", valid_accuracy)
  elapsed_epoch_time = (time.time() - start_time) / 60
  print("Epoch Elapsed Time: %d mins" % (elapsed_epoch_time))
total_training_time = (time.time() - start_time) / 60
print("Total Training Time: %d mins" % (elapsed_epoch_time))
