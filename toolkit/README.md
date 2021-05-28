## Commonly used toolkits

## Create Text-Classification Dataset and DataLoader using GLUE(SST2)

Only creates a PyTorch Dataset:

```python
from transformers import T5Tokenizer
from custom_dataset_nlp import DatasetTextClassification

tokenizer = T5Tokenizer.from_pretrained('t5-small')
dataset = DatasetTextClassification(tokenizer,
				    max_input_length=32,
				    train=True,
				    )

```

Creates a DataLoader while implicitly creating a TextClassificationDataset

```python
from transformers import T5Tokenizer
from custom_dataset_nlp import DataLoaderTextClassification

tokenizer = T5Tokenizer.from_pretrained('t5-small')
train_loader = DataLoaderTextClassification(tokenizer, max_input_length=64, train=True).return_dataloader(batch_size=32, shuffle=True)
```
