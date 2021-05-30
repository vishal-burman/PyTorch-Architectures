import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer
from datasets import load_dataset
from sampler import SortishSampler

class DatasetTextClassification(Dataset):
    def __init__(self, tokenizer, max_input_length=16, train=True):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.train = train
        self.dataset = load_dataset('glue', 'sst2')
        if self.train:
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
        return {
                'sents': sentences,
                'labels': targets,
                }

    def collate_fn(self, batch):
        sentences = []
        labels = []
        for sample in batch:
            sentences.append(sample['sents'])
            labels.append(sample['labels'])
        tokens = self.tokenizer(sentences,
                max_length=self.max_input_length,
                padding=True,
                truncation=True,
                return_tensors='pt')
        return{
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'labels': torch.tensor(labels),
                }

class DatasetLanguageModeling(Dataset):
    def __init__(self, tokenizer, max_input_length=16, train=True, split=None):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, XLNetTokenizer):
            if self.max_input_length % 2 != 0:
                raise ValueError('Use even lengths for XLNet Model')
        self.max_input_length = max_input_length
        self.dataset = load_dataset('wikitext', 'wikitext-103-v1')
        if self.train:
            self.sents = self.dataset['train']['text']
        else:
            self.sents = self.dataset['validation']['text']
        if split is not None:
            self.sents = self.sents[:split]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        sents = self.sents[idx]
        return {
                'sents': sents,
                }

    def collate_fn(self):
        pass


class DataLoaderTextClassification:
    def __init__(self, tokenizer, max_input_length=16, train=True):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.train = train
        self.dataset = DatasetTextClassification(self.tokenizer, self.max_input_length, self.train)

    def return_dataloader(self, batch_size=4, shuffle=False, sortish_sampler=False):
        if sortish_sampler:
            src_lens = [len(x) for x in self.dataset.sents]
            sampler = SortishSampler(src_lens, batch_size, shuffle=shuffle)
            return DataLoader(self.dataset, batch_size, collate_fn=self.dataset.collate_fn, sampler=sampler)
        return DataLoader(self.dataset, batch_size, shuffle=shuffle, collate_fn=self.dataset.collate_fn)
