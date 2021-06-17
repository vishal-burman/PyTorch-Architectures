import os
import logging
import urllib
import tarfile
import string
import gc
import torch
from torch.utils.data import Dataset, DataLoader
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
    return sents, labels

def get_language_modeling_dataset(train=True, hf=True):
    if hf:
        dataset = load_dataset('wikitext', 'wikitext-103-v1')
        sents = dataset[('train' if train else 'validation')]['text']
    else:
        if os.path.exists(os.path.join(os.getcwd(), 'wikitext-103')):
            logging.warn('wikitext-103 exists...')
        else:
            logging.warn('Manual download from https://course.fastai/datasets')
            urllib.request.urlretrieve('https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz', 'wikitext-103.tgz')
            print('wikitext-103 downloaded...')
            tf = tarfile.open('wikitext-103.tgz')
            tf.extractall(path='.')
            print('wikitext-103.tgz extracted...')
        total_sents = open(os.path.join(os.getcwd(), 'wikitext-103', ('train.csv' if train else 'test.csv'))).readlines()
        split = 80 * len(total_sents) // 100
        logging.warn('Using an 80% train and 20% validation split')
        train_sents = total_sents[:split]
        valid_sents = total_sents[split:]
        assert len(train_sents) + len(valid_sents) == len(total_sents), 'Split not successful'
        sents = train_sents if train else valid_sents
    return sents

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

def is_cuda_out_of_memory(exception):
    return isinstance(exception, RuntimeError) \
            and len(exception.args) == 1 \
            and "CUDA" in exception.args[0] \
            and "out of memory" in exception.args[0]

def is_cudnn_snafu(exception):
    return isinstance(exception, RuntimeError) \
            and len(exception.args) == 1 \
            and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]

def is_out_of_cpu_memory(exception):
    return isinstance(exception, RuntimeError) \
            and len(exception.args) == 1 \
            and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]

def is_oom_error(exception):
    return is_cuda_out_of_memory(exception) \
            or is_cudnn_snafu(exception) \
            or is_out_of_cpu_memory(exception)

def gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        try: # Last thing which should cause OOM error but seemingly it can
            torch.cuda.empty_cache()
        except:
            if not is_oom_error(exception):
                raise

def get_optimal_batchsize(dataset, model, max_trials=25):
    device = torch.device('cuda:0' if torch.cuda.is_available() \
                           else 'cpu')
    model.to(device)
    bs = 1
    dataloader = DataLoader(dataset, batch_size=bs)
    for _ in max_trials:
        gc_cuda()
        try:
            sample = next(iter(dataloader))
            
            if type(sample) is dict:
                sample = dict_to_device(sample, device)
                outputs = model(**sample)
            elif hasattr(sample, 'data'):
                sample = dict_to_device(sample.data, device)
                outputs = model(**sample)
            else:
                raise NotImplementedError('DataLoader should yield dict or BatchEncoding types')

            bs = int(bs * 2.0)
            dataloader = DataLoader(dataset, batch_size=bs)
        except RuntimeError as exception:
            if is_oom_error(exception):
                gc_cuda()
                bs = int(bs * 0.5)
                dataloader = DataLoader(dataset, batch_size=bs)
                break
            else:
                raise # some other error not memory related
        return bs
