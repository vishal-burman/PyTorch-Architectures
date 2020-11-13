import torch
import torch.nn as nn
from model import GPT2Classify

model_gpt2 = GPT2Classify()
input_ids, attention_mask = torch.ones((1, 8), dtype=torch.long), torch.ones((1, 8), dtype=torch.long)
labels = torch.tensor([1])
outputs = model_gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
total_params = sum(p.numel() for p in model_gpt2.parameters())
print("Total Parameters = ", total_params)
print("Done")
