import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, XLNetTokenizer

from .utils import get_classification_dataset, get_language_modeling_dataset


class DatasetTextClassification(Dataset):
    def __init__(self, tokenizer, max_input_length=16, train=True, split=None):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.sents, self.labels = get_classification_dataset(train, split)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        sentences, targets = self.sents[idx], self.labels[idx]
        return {
            "sents": sentences,
            "labels": targets,
        }

    def collate_fn(self, batch):
        sentences = []
        labels = []
        for sample in batch:
            sentences.append(sample["sents"])
            labels.append(sample["labels"])
        tokens = self.tokenizer(
            sentences,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": torch.tensor(labels),
        }


class DatasetCausalLanguageModeling(Dataset):
    def __init__(
        self, tokenizer, input_texts=None, max_input_length=16, train=True, hf=True
    ):
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
            return_tensors="pt",
        )
        labels = tokens["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        tokens["labels"] = labels
        return tokens


class DatasetMaskedLanguageModeling(Dataset):
    def __init__(
        self,
        tokenizer,
        input_texts=None,
        max_input_length=16,
        train=True,
        mlm=0.15,
        hf=True,
    ):
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
        tokens = self.tokenizer(
            sentences,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if self.mlm_probability is not None:
            tokens["input_ids"], tokens["labels"] = self.mask_tokens_mlm(tokens)
        return tokens

    def mask_tokens_mlm(self, tokens):
        input_ids = tokens["input_ids"]
        special_tokens_mask = tokens.pop("special_tokens_mask", None)
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of time, we replace masked input tokens with [MASK] token
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # The  rest of the 10% we keep the masked input tokens unchanged
        return input_ids, labels


class DatasetPermutationLanguageModeling(Dataset):
    def __init__(
        self,
        tokenizer,
        input_texts=None,
        max_input_length=16,
        train=True,
        plm=1 / 6,
        max_span_length=5,
        hf=True,
    ):
        self.tokenizer = tokenizer
        self.plm_probability = plm
        if max_input_length % 2 != 0:
            raise ValueError("To prevent leakage use even-length")
        self.max_input_length = max_input_length
        if input_texts is not None:
            self.sents = input_texts
        else:
            self.sents = get_language_modeling_dataset(train, hf)
        self.plm = plm
        self.max_span_length = max_span_length

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
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if self.plm_probability is not None:
            (
                tokens["input_ids"],
                tokens["perm_mask"],
                tokens["target_mapping"],
                tokens["labels"],
            ) = self.mask_tokens_plm(tokens)
        return tokens

    def mask_tokens_plm(self, tokens):
        """
        Algorithm:
        1.
        2.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "mask token is necessary for Permutation Language Modeling"
            )
        input_ids = tokens["input_ids"]
        labels = input_ids.clone()

        masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)
        target_mapping = torch.zeros(
            (labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32
        )

        for i in range(labels.size(0)):
            cur_len = 0
            max_len = labels.size(1)

            while cur_len < max_len:
                span_length = torch.randint(1, self.max_span_length + 1, (1,)).item()
                context_length = int(span_length / self.plm_probability)
                start_index = (
                    cur_len
                    + torch.randint(context_length - span_length + 1, (1,)).item()
                )
                masked_indices[i, start_index : start_index + span_length] = 1
                cur_len += context_length
            target_mapping[i] = torch.eye(labels.size(1))

        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ],
            dtype=torch.bool,
        )
        masked_indices.masked_fill_(special_tokens_mask, value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            masked_indices.masked_fill_(padding_mask, value=0.0)

        non_func_mask = ~(padding_mask | special_tokens_mask)
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        perm_mask = torch.zeros(
            (labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32
        )
        for i in range(labels.size(0)):
            perm_index = torch.arange(labels.size(1))
            perm_index = perm_index.reshape((-1, labels.size(1) // 2)).transpose(0, 1)
            perm_index = perm_index[torch.randperm(labels.size(1) // 2)]
            perm_index = torch.flatten(perm_index.transpose(0, 1))
            perm_index.masked_fill_(~masked_indices[i] & non_func_mask[i], -1)
            perm_mask[i] = (
                perm_index.reshape((labels.size(1), 1))
                <= perm_index.reshape((1, labels.size(1)))
            ) & masked_indices[i]
        return input_ids.long(), perm_mask, target_mapping, labels.long()


class DatasetT5MaskedLanguageModeling(Dataset):
    def __init__(self, tokenizer: T5Tokenizer, input_texts):
        self.tokenizer = tokenizer
        self.input_texts = input_texts

    def __len__(
        self,
    ):
        return len(self.input_texts)

    def __getitem__(self, idx):
        return self.input_texts[idx]

    def collate_fn(
        self,
    ):
        pass


class DataLoaderTextClassification:
    def __init__(self, tokenizer, max_input_length=16, train=True, split=None):
        self.dataset = DatasetTextClassification(
            tokenizer, max_input_length, train, split
        )

    def return_dataloader(self, batch_size=4, shuffle=False, sampler=None):
        if sampler is not None:
            return DataLoader(
                self.dataset,
                batch_size,
                collate_fn=self.dataset.collate_fn,
                sampler=sampler,
            )
        return DataLoader(
            self.dataset,
            batch_size,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_fn,
        )


class DataLoaderCausalLanguageModeling:
    def __init__(
        self, tokenizer, input_texts=None, max_input_length=16, train=True, hf=True
    ):
        self.dataset = DatasetCausalLanguageModeling(
            tokenizer=tokenizer,
            input_texts=input_texts,
            max_input_length=max_input_length,
            train=train,
            hf=hf,
        )

    def return_dataloader(self, batch_size=4, shuffle=False):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_fn,
        )


class DataLoaderMaskedLanguageModeling:
    def __init__(
        self,
        tokenizer,
        input_texts=None,
        max_input_length=16,
        train=True,
        mlm=0.15,
        hf=True,
    ):
        self.dataset = DatasetMaskedLanguageModeling(
            tokenizer=tokenizer,
            input_texts=input_texts,
            max_input_length=max_input_length,
            train=train,
            mlm=mlm,
            hf=hf,
        )

    def return_dataloader(self, batch_size=4, shuffle=False):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_fn,
        )


class DataLoaderPermutationLanguageModeling:
    def __init__(
        self,
        tokenizer,
        input_texts=None,
        max_input_length=16,
        train=True,
        plm=1 / 6,
        max_span_length=5,
        hf=True,
    ):
        self.dataset = DatasetPermutationLanguageModeling(
            tokenizer=tokenizer,
            input_texts=input_texts,
            max_input_length=max_input_length,
            train=train,
            plm=plm,
            max_span_length=max_span_length,
            hf=hf,
        )

    def return_dataloader(self, batch_size=4, shuffle=False):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_fn,
        )
