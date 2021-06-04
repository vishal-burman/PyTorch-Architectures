import os
import urllib
import tarfile
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer
import datasets
from datasets import load_dataset

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
    def __init__(self, tokenizer, input_texts=None, max_input_length=16, train=True, split=None, mlm=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm
        if isinstance(self.tokenizer, XLNetTokenizer):
            if max_input_length % 2 != 0:
                raise ValueError('Use even lengths for XLNet Model')
        self.max_input_length = max_input_length
        if input_texts is not None:
            self.dataset = input_texts
        else:
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
            if input_texts is not None and not train:
                self.sents = self.sents[split:]
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
                                padding=('max_length' if isinstance(self.tokenizer, XLNetTokenizer) else True),
                                truncation=True,
                                return_tensors='pt')
        if self.mlm_probability is not None:
            tokens['input_ids'], tokens['target_ids'] = self.mask_tokens_mlm(tokens)
        return tokens

    def mask_tokens_mlm(self, tokens):
        input_ids = tokens['input_ids']
        special_tokens_mask = tokens.pop('special_tokens_mask', None)
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                    ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100 # Only compute loss on masked tokens

        # 80% of time, we replace masked input tokens with [MASK] token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The  rest of the 10% we keep the masked input tokens unchanged
        return input_ids, labels


class DataLoaderTextClassification:
    def __init__(self, tokenizer, max_input_length=16, train=True):
        self.dataset = DatasetTextClassification(tokenizer, max_input_length, train)

    def return_dataloader(self, batch_size=4, shuffle=False, sampler=None):
        if sampler is not None:
            return DataLoader(self.dataset, batch_size, collate_fn=self.dataset.collate_fn, sampler=sampler)
        return DataLoader(self.dataset, batch_size, shuffle=shuffle, collate_fn=self.dataset.collate_fn)

class DataLoaderLanguageModeling:
    def __init__(self, tokenizer, input_texts=None, max_input_length=16, train=True, split=None):
        self.dataset = DatasetLanguageModeling(tokenizer, input_texts, max_input_length, train, split)

    def return_dataloader(self, batch_size=4, shuffle=False):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.dataset.collate_fn)
