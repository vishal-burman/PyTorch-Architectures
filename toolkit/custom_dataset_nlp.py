import os
import urllib
import tarfile
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer
import datasets
from datasets import load_dataset
from .sampler import SortishSampler
from .utils import remove_punctuation

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
            if max_input_length % 2 != 0:
                raise ValueError('Use even lengths for XLNet Model')
        self.max_input_length = max_input_length
        try:
            self.dataset = load_dataset('wikitext', 'wikitext-103-v1')
        except:
            if os.path.exists(os.path.join(os.getcwd(), 'wikitext-103')):
                print('wikitext-103 exists...')
                self.dataset = open(os.path.join(os.getcwd(), 'wikitext-103', ('train.csv' if train else 'test.csv'))).readlines()
            else:
                warnings.warn('Manual download from https://course.fastai/datasets')
                urllib.request.urlretrieve('https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz', 'wikitext-103.tgz')
                print('wikitext-103.tgz downloaded...')
                tf = tarfile.open('wikitext-103.tgz')
                tf.extractall(path='.')
                print('wikitext-103.tgz extracted...')
                self.dataset = open(os.path.join(os.getcwd(), 'wikitext-103', ('train.csv' if train else 'test.csv'))).readlines()
        if isinstance(self.dataset, datasets.dataset_dict.DatasetDict):
            self.sents = self.dataset[('train' if train else 'validation')]['text']
        else:
            self.sents = self.dataset

        if split is not None:
            self.sents = self.sents[:split]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]

    def collate_fn(self, batch):
        sentences = []
        for sentence in batch:
            sentences.append(sentence)
        tokens = self.tokenizer(sentences,
                                max_length=self.max_input_length,
                                padding='max_length' if isinstance(self.tokenizer, XLNetTokenizer) else True,
                                truncation=True,
                                return_tensors='pt')
        return tokens

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
