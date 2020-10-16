########################################
# This file is used to debug module
# IGNORE THIS
########################################

import torch
from transformers import XLMTokenizer
from model import XLMModel
from config_xlm import XLMConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = XLMConfig()
tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')

input_text = "This is a very good model"

tokens = tokenizer(input_text, max_length=8, truncation=True, padding='max_length')
ids = torch.tensor(tokens['input_ids'], dtype=torch.long).unsqueeze(0)
mask = torch.tensor(tokens['attention_mask'], dtype=torch.long).unsqueeze(0)
model = XLMModel(config).to(device)

output = model(input_ids=ids.to(device), attention_mask=mask.to(device))
print(output.shape)
