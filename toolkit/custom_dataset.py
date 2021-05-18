import torch
from torch.utils.data import Dataset
from datasets import load_dataset
class DatasetTextClassification(Dataset):
    def __init__(self, tokenizer, max_input_length=16, split=0.9, train=True):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.dataset = load_dataset('glue', 'sst2')
        if train:
            self.sents = self.dataset['train']['sentence']
            self.labels = self.dataset['train']['label']
        else:
            self.sents = self.dataset['validation']['sentence']
            self.labels = self.dataset['validation']['label']
        assert len(self.sents) == len(self.labels), 'Size of sentences and labels do not match'

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        sentences, targets = self.sents[idx], self.labels[idx]
        tokens = self.tokenizer(sentences, max_length=self.max_input_length, padding='max_length', truncation=True, return_tensors='pt')
        return {
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'labels': torch.tensor(targets),
                }
