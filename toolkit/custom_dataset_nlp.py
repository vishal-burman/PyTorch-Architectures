import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer
import datasets
from .utils import get_classification_dataset, get_language_modeling_dataset

class DatasetTextClassification(Dataset):
    def __init__(self, tokenizer, max_input_length=16, train=True, split=None):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        sents, labels = get_classification_dataset(train, split)

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

class DatasetCausalLanguageModeling(Dataset):
    def __init__(self, tokenizer, input_texts=None, max_input_length=16, train=True, hf=True):
        self.tokenizer = tokenizer
        if input_texts is not None:
            self.sents = input_texts
        else:
            self.sents = get_language_modeling_dataset(train, hf)
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]

    def collate_fn(self, batch):
        sentences = []
        for sentence in batch:
            sentences.append(sentence)
        tokens = self.tokenizer(
                sentences,
                max_length=self.max_input_length,
                padding=True,
                truncation=True,
                return_tensors='pt',
                )
        labels = tokens['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        tokens['labels'] = labels
        return tokens

class DatasetMaskedLanguageModeling(Dataset):
    def __init__(self, tokenizer, input_texts=None, max_input_length=16, train=True, mlm=0.15, hf=True):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm
        self.max_input_length = max_input_length
        if input_texts is not None:
            self.sents = input_texts
        else:
            self.sents = get_language_modeling_dataset(train, hf)

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
                                padding=True,
                                truncation=True,
                                return_tensors='pt')
        if self.mlm_probability is not None:
            tokens['input_ids'], tokens['labels'] = self.mask_tokens_mlm(tokens)
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

class DatasetPermutationLanguageModeling(Dataset):
    def __init__(self, tokenizer, input_texts=None, max_input_length=16, train=True, plm=1/6, hf=True):
        self.tokenizer = tokenizer
        self.plm_probability = plm
        if max_input_length % 2 != 0:
            raise ValueError('To prevent leakage use even-length')
        self.max_input_length = max_input_length
        if input_texts is not None:
            self.sents = input_texts
        else:
            self.sents = get_language_modeling_dataset(train, hf)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]

    def collate_fn(self, batch):
        sentences = []
        for sentence in batch:
            sentences.append(sentence)
        tokens = self.tokenizer(
                sentences,
                max_length=self.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                )
        if self.plm_probability is not None:
            tokens['input_ids'], tokens['perm_mask'], tokens['target_mapping'], tokens['labels'] = self.mask_tokens_plm(tokens)
        return tokens

    def mask_tokens_plm(self, tokens):
        raise NotImplementedError


class DataLoaderTextClassification:
    def __init__(self, tokenizer, max_input_length=16, train=True, split=None):
        self.dataset = DatasetTextClassification(tokenizer, max_input_length, train, split)

    def return_dataloader(self, batch_size=4, shuffle=False, sampler=None):
        if sampler is not None:
            return DataLoader(self.dataset, batch_size, collate_fn=self.dataset.collate_fn, sampler=sampler)
        return DataLoader(self.dataset, batch_size, shuffle=shuffle, collate_fn=self.dataset.collate_fn)

class DataLoaderMaskedLanguageModeling:
    def __init__(self, tokenizer, input_texts=None, max_input_length=16, train=True, mlm=0.15, hf=True):
        self.dataset = DatasetLanguageModeling(
                tokenizer=tokenizer,
                input_texts=input_texts, 
                max_input_length=max_input_length, 
                train=train,
                mlm=mlm,
                hf=hf,
                )

    def return_dataloader(self, batch_size=4, shuffle=False):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.dataset.collate_fn)
