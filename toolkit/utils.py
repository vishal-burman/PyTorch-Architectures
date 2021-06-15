import string
import torch
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset

def get_classification_dataset(train=True, split=None):
    dataset = load_dataset('glue', 'sst2')
    if train:
        sents = dataset['train']['sentence']
        labels = dataset['train']['label']
    else:
        sents = dataset['validation']['sentence']
        labels = dataset['validation']['label']
    assert len(sents) == len(labels), 'Input and Output shape do not match'
    if split is not None:
        sents = sents[:split]
        labels = labels[:split]
    return (sents, labels)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def dict_to_device(sample_dict, device=torch.device('cpu')):
    keys, values = list(sample_dict.keys()), list(sample_dict.values())
    if not all(isinstance(x, torch.Tensor) for x in values):
        raise TypeError('Only torch.Tensor values can be shifted to CUDA')
    values = list(map(lambda x: x.to(device), values))
    final_dict = dict(zip(keys, values))
    return final_dict
