import torch
import torch.nn as nn


def init_optimizer(optimizer: str, **kwargs):
    optimizer = optimizer.lower()
    if optimizer == "adam":
        return torch.optim.Adam(**kwargs)
    elif optimizer == "adamw":
        return torch.optim.AdamW(**kwargs)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
    ):
        self.model = model
        self.optimizer = init_optimizer(optimizer)

    def train(
        self,
    ):
        pass
