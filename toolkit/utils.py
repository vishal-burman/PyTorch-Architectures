import copy
import os
import logging
import urllib
import tarfile
import string
import wget
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.onnx
from datasets import load_dataset
from .cuda_memory_utils import gc_cuda, is_oom_error


def get_classification_dataset(train=True, split=None):
    dataset = load_dataset("glue", "sst2")
    if train:
        sents = dataset["train"]["sentence"]
        labels = dataset["train"]["label"]
    else:
        sents = dataset["validation"]["sentence"]
        labels = dataset["validation"]["label"]
    assert len(sents) == len(labels), "Input and Output shape do not match"
    if split is not None:
        sents = sents[:split]
        labels = labels[:split]
    return sents, labels


def get_language_modeling_dataset(train=True, hf=True):
    if hf:
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        sents = dataset[("train" if train else "validation")]["text"]
    else:
        if os.path.exists(os.path.join(os.getcwd(), "wikitext-103")):
            logging.warn("wikitext-103 exists...")
        else:
            logging.warn("Manual download from https://course.fastai/datasets")
            urllib.request.urlretrieve(
                "https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz",
                "wikitext-103.tgz",
            )
            print("wikitext-103 downloaded...")
            tf = tarfile.open("wikitext-103.tgz")
            tf.extractall(path=".")
            print("wikitext-103.tgz extracted...")
        total_sents = open(
            os.path.join(
                os.getcwd(), "wikitext-103", ("train.csv" if train else "test.csv")
            )
        ).readlines()
        split = 80 * len(total_sents) // 100
        logging.warn("Using an 80% train and 20% validation split")
        train_sents = total_sents[:split]
        valid_sents = total_sents[split:]
        assert len(train_sents) + len(valid_sents) == len(
            total_sents
        ), "Split not successful"
        sents = train_sents if train else valid_sents
    return sents


def get_image_classification_dataset(
    train=True,
):
    if os.path.exists(os.path.join(os.getcwd(), "cifar10")):
        print("cifar10 exists...")
    else:
        filename = wget.download(
            "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        )
        tf = tarfile.open("cifar10.tgz")
        tf.extractall(path=".")
        print("cifar10 extracted...")

    parent_path = os.path.join(os.getcwd(), "cifar10", ("train" if train else "test"))
    labels = os.listdir(parent_path)
    final_list = []
    for idx, label in enumerate(labels):
        path = os.path.join(parent_path, label)
        images_paths = [os.path.join(path, f) for f in os.listdir(path)]
        tuple_list = [(image_path, idx) for image_path in images_paths]
        final_list.extend(tuple_list)
    return final_list


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def remove_punctuation(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def dict_to_device(sample_dict, device=torch.device("cpu")):
    keys, values = list(sample_dict.keys()), list(sample_dict.values())
    if not all(isinstance(x, torch.Tensor) for x in values):
        raise TypeError("Only torch.Tensor values can be shifted to CUDA")
    values = list(map(lambda x: x.to(device), values))
    final_dict = dict(zip(keys, values))
    return final_dict


def tuple_to_device(sample_tuple, device=torch.device("cpu")):
    assert len(sample_tuple) == 2, "Tuple should be of the format (Inputs, Labels)"
    if not all(isinstance(x, torch.Tensor) for x in sample_tuple):
        raise TypeError("Only torch.Tensor values can be shifted to CUDA")
    new_tuple = tuple(map(lambda x: x.to(device), sample_tuple))
    return new_tuple


def _trial_run(model, dataloader, device, step_limit=3):
    model_copy = copy.deepcopy(model)
    model_copy.to(device)

    for idx, sample in enumerate(dataloader):
        if idx >= step_limit:
            break

        if type(sample) is dict:
            sample = dict_to_device(sample, device)
            outputs = model_copy(**sample)
        elif type(sample) is tuple:
            sample = tuple_to_device(sample, device)
            outputs = model_copy(sample[0], sample[1])
        elif hasattr(sample, "data"):
            sample = dict_to_device(sample.data, device)
            outputs = model_copy(**sample)
        else:
            raise ValueError("DataLoader should yield dict or BatchEncoding types")

    del model_copy


def _run_power_scaling(model, dataset, max_trials):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bs = 1
    dataloader = DataLoader(
        dataset, batch_size=bs, shuffle=True, collate_fn=dataset.collate_fn
    )
    for _ in range(max_trials):
        gc_cuda()
        try:
            _trial_run(model, dataloader, device)

            bs = int(bs * 2.0)
            dataloader = DataLoader(
                dataset, batch_size=bs, shuffle=True, collate_fn=dataset.collate_fn
            )
        except RuntimeError as exception:
            if is_oom_error(exception):
                gc_cuda()
                bs = int(bs * 0.5)
                dataloader = DataLoader(
                    dataset, batch_size=bs, shuffle=True, collate_fn=dataset.collate_fn
                )
                break
            else:
                raise  # some other error not memory related
    return bs


def get_optimal_batchsize(dataset, model, max_trials=25):
    if not hasattr(dataset, "collate_fn"):
        raise AttributeError(
            "Define a collate_fn in your Dataset and make sure it returns dict type"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    bs = _run_power_scaling(model, dataset, max_trials)
    return bs


def image_interpolation(image: torch.Tensor, target_size):
    batch, channels, height, width = image.size()
    assert height == width, "Height and Width of the image is not equal"

    if type(target_size) is tuple:
        target_height, target_width = target_size[0], target_size[1]
    elif type(target_size) is int:
        target_height, target_width = target_size, target_size
    else:
        raise TypeError("target_size should either be int or tuple")
    assert target_height == target_width, "Target Height and Target Width is not equal"

    if height < target_height:
        image = F.interpolate(
            image, size=target_height, mode="bilinear", align_corners=True
        )
    else:
        image = F.interpolate(image, size=target_height, mode="nearest")

    return image


def convert_to_onnx(torch_model, sample_input: torch.Tensor, save_path: str):
    """Converts a torch module to onnx"""
    if torch_model.training:
        logging.warn("Model is in training mode...switching to eval mode")
        torch_model.eval()

    assert sample_input.requires_grad, "Needs sample_input's requires_grad to be True"

    raise NotImplementedError("TBD")


class EarlyStopping:
    def __init__(
        self, metric, patience=3, verbose=False, delta=0, path="checkpoint.pt"
    ):
        assert (
            metric == "val_loss" or metric == "val_accuracy"
        ), "metric should either be val_loss or val_accuracy"
        self.metric = metric
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = -np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, item, model):
        if type(item) is torch.Tensor:
            item = item.item()

        if model.training:
            print("Model is in train mode...switching to eval mode")
            model.eval()

        if self.metric == "val_loss":
            self._early_stop_loss(item, model)
        else:
            self._early_stop_accuracy(item, model)

    def _early_stop_accuracy(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def _early_stop_loss(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, item, model):
        """Saves model when validation loss decrease"""
        if self.metric == "val_loss":
            if self.verbose:
                print(
                    f"Validation loss decreased from {self.val_loss_min:.3f} to {item:.3f}"
                )
            self.val_loss_min = item
        else:
            if self.verbose:
                print(
                    f"Validation accuracy increased from {self.val_acc_max:.2f}% to {item:.2f}%"
                )
            self.val_acc_max = item
        torch.save(model.state_dict(), self.path)
