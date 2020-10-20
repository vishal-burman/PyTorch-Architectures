########################################
# This file is used to debug module
# IGNORE THIS
########################################

from os import path
import sys
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import XLMTokenizer
from model import XLMWithLMHeadModel
from config_xlm import XLMConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = XLMConfig()
config.n_layers = 1
tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
model = XLMWithLMHeadModel(config).to(device)

########################################################################
# Dataset
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

        return{
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(ids, dtype=torch.long),
                }

    def build(self):
        for t in self.texts:
            self.train_list.append(tokenizer(t, max_length=64, pad_to_max_length=True, truncation=True))
#########################################################################
if path.exists('train_dataset.pt') and path.exists('valid_dataset.pt'):
    print("Saved PyTorch dataset exists!")
    train_dataset = torch.load('train_dataset.pt')
    valid_dataset = torch.load('valid_dataset.pt')
else:
    texts = []
    labels = []
    with open("dataset.csv", "r") as file_1:
        reader = csv.reader(file_1)
        for line in reader:
            texts.append(line[0].strip())
            labels.append(line[1].strip())

    texts = texts[1:]

    texts_train = texts[:10000]

    texts_valid = texts[10000:10100]

    start_time = time.time()
    train_dataset = CustomDataset(texts_train, tokenizer)
    valid_dataset = CustomDataset(texts_valid, tokenizer)
    print("Dataset Conversion Done!!")
    print("Time Taken = ", (time.time() - start_time)/60)
    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(valid_dataset, 'valid_dataset.pt')
#########################################################################

# Hyperparameters

BATCH_SIZE = 4
LR = 5e-04
EPOCHS = 10

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=BATCH_SIZE)
print("Length of Train DataLoader: ", len(train_loader))
print("Length of Valid DataLoader: ", len(valid_loader))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

############################################################################
def compute_loss(model, data_loader, device):
  total_loss = 0
  model.eval()
  with torch.set_grad_enabled(False):
    for sample in data_loader:
      ids = sample['ids'].to(device)
      mask = sample['mask'].to(device)
      labels = sample['ids'].to(device)

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
    labels = sample['ids'].to(device)
    
    optimizer.zero_grad()
    
    logits = model(input_ids=ids, attention_mask=mask, labels=labels)
    loss = logits[0]
    print(logits[1].shape)
    sys.exit("Break")

    # LOGGING
    if idx % 100 == 0:
      print("Batch: %04d/%04d || Epoch: %03d/%03d" % (idx, len(train_loader), epoch+1, EPOCHS))

    loss.backward()
    optimizer.step()

  model.eval()
  with torch.set_grad_enabled(False):
    train_loss = compute_loss(model, train_loader, device)
    valid_loss = compute_loss(model, valid_loader, device)
    print("Train Loss: %.3f" % (train_loss))
    print("Valid Loss: %.3f" % (valid_loss))
  elapsed_epoch_time = (time.time() - start_time) / 60
  print("Epoch Elapsed Time: %d mins" % (elapsed_epoch_time))
total_training_time = (time.time() - start_time) / 60
print("Total Training Time: %d mins" % (elapsed_epoch_time))
