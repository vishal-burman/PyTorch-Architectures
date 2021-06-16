# Commonly used toolkits


## Only creates a PyTorch Dataset:

```python
from transformers import T5Tokenizer
from custom_dataset_nlp import DatasetTextClassification

tokenizer = T5Tokenizer.from_pretrained('t5-small')
dataset = DatasetTextClassification(tokenizer,
				    max_input_length=32,
				    train=True,
				    )

```

## Creates a DataLoader while implicitly creating a TextClassificationDataset

```python
from transformers import T5Tokenizer
from custom_dataset_nlp import DataLoaderTextClassification

tokenizer = T5Tokenizer.from_pretrained('t5-small')
train_loader = DataLoaderTextClassification(tokenizer, max_input_length=64, train=True).return_dataloader(batch_size=32, shuffle=True)
```

## Create a SortishSampler while implicitly creating a TextClassificationDataset

```python
from transformers import T5Tokenizer
from custom_dataset_nlp import DataLoaderTextClassification

tokenizer = T5Tokenizer.from_pretrained('t5-small') 
train_loader = DataLoaderTextClassification(tokenizer, max_input_length=64, train=True).return_dataloader(batch_size=32, sortish_sampler=True)
```

## Create an MLM(Masked Language Modeling) objective from input texts

```python
from transformers import DistilBertTokenizer
from toolkit.custom_dataset_nlp import DataLoaderMaskedLanguageModeling

input_texts = ['The quick brown fox jumped over the lazy dog']
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

loader = DataLoaderMaskedLanguageModeling(tokenizer, input_texts=input_texts)
loader = loader.return_dataloader(batch_size=1)
sample = next(iter(loader))
print('Input --> ', tokenizer.decode(sample['input_ids'][0]))
print('Output --> ', tokenizer.decode(sample['labels'][0]))

# Input -->  [CLS] [MASK] [MASK] brown fox jumped over the lazy dog [SEP]
# Output -->  [UNK] the quick [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK]
```

## Create a PLM(Permutation Language Modeling) objective from input texts

```python
from transformers import XLNetTokenizer
from toolkit.custom_dataset_nlp import DataLoaderPermutationLanguageModeling

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
input_texts = ['The quick brown fox jumped over the lazy dog']

loader = DataLoaderPermutationLanguageModeling(tokenizer, input_texts=input_texts)
loader = loader.return_dataloader(batch_size=1)
sample = next(iter(loader))
input_ids = sample['input_ids'] # [bs, max_len]
perm_mask = sample['perm_mask'] # [bs, max_len, max_len]
target_mapping = sample['target_mapping'] # [bs, max_len, max_len]
labels = sample['labels'] # [bs, max_len]
```

## Function to calculate accuracy for NLP Tasks

```python
import torch
from model import YourModel
from custom_dataset_nlp import DataLoaderTextClassification

model = YourModel(**parameter_dict)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoaderTextClassification(tokenizer, max_input_length=64, train=True).return_dataloader(batch_size=32, sortish_sampler=True)

acc = nlp_compute_accuracy(model, train_loader, device)
```
