import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .metrics import cv_compute_accuracy, nlp_compute_accuracy, nlp_compute_mean_loss
from .utils import dict_to_device, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)


def plot_grad_flow(layers_name: List[str], average_gradients: List[torch.Tensor]):
    """
    Plots the gradient flow in each layer with each epoch
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
    """
    plt.plot(average_gradients, alpha=0.3, color="b")
    plt.hlines(0, 0, len(average_gradients) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(average_gradients) + 1), layers_name, rotation="vertical")
    plt.xlim(xmin=0, xman=len(average_gradients))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradients")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.show()


def init_optimizer(optimizer: str, model: nn.Module, lr: float):
    """
    Maps optimizer to its corresponging torch function
    """
    optimizer = optimizer.lower()
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Union[Dataset, DataLoader],
        valid_dataset: Union[Dataset, DataLoader],
        fp16: bool = False,
    ):
        self.model = model

        assert type(train_dataset) == type(
            valid_dataset
        ), f"train_dataset is {type(train_dataset)} and valid_dataset is {type(valid_dataset)}"
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if fp16:
            assert (
                torch.cuda.is_available()
            ), f"fp16 available only for CUDA devices, found {self.device}"
            raise NotImplementedError  # TODO

    def train(
        self,
        optimizer: str = "adam",
        scheduler: Optional[str] = "linear",
        lr: float = 3e-5,
        epochs: int = 3,
        batch_size: int = 32,
        shuffle_train: bool = False,
        shuffe_valid: bool = False,
        num_warmup_steps: int = 0,
        metric: str = "nlp_perplexity",
        show_grad_flow: str = False,
    ):
        if not self.model.training:
            self.model.train()

        if type(self.train_dataset) is Dataset:
            train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=shuffle_train
            )
            valid_loader = DataLoader(
                self.valid_dataset, batch_size=batch_size, shuffle=shuffle_valid
            )
        elif type(self.train_dataset) is DataLoader:
            train_loader = self.train_dataset
            valid_loader = self.valid_dataset
        else:
            raise NotImplementedError

        num_training_steps = epochs * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))
        optimizer = init_optimizer(optimizer, self.model, lr)
        if scheduler is not None:
            scheduler_func = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=-1,
            )

        # Details
        logging.info("********** Running Training **********")
        logging.info(f"  Total Training Steps = {num_training_steps}  ")
        logging.info(f"  Epochs = {epochs}  ")
        logging.info(f"  Batch Size = {batch_size}  ")
        logging.info(f"  Length of Train DataLoader = {len(train_loader)}  ")
        logging.info(f"  Length of Valid DataLoader = {len(valid_loader)}  ")

        for epoch in range(epochs):
            loss_list = []
            layers = []
            average_gradients = []
            if not self.model.training:
                self.model.train()

            for idx, sample in enumerate(train_loader):
                loss, logits = self.model(**dict_to_device(sample, device=self.device))
                loss_list.append(loss.item())
                loss.backward()
                if show_grad_flow:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and "bias" not in name:
                            layers.append(name)
                            average_gradients.append(param.grad.abs().mean())

                optimizer.step()
                if scheduler is not None:
                    scheduler_func.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            self.model.eval()
            with torch.set_grad_enabled(False):
                metric_output = self.validate(valid_loader, metric=metric)

            mean_loss = torch.mean(torch.tensor(loss_list)).item()
            logging.info(
                f"Epoch: {epoch + 1} || Training Loss: {mean_loss:.3f} || {metric}: {metric_output:.3f}"
            )
            logging.info(f"Gradient-Flow for epoch {epoch + 1}")
            plot_grad_flow(layers, average_gradients)

    def validate(
        self,
        dataloader,
        metric,
    ):
        if self.model.training:
            self.model.eval()

        if metric == "nlp_accuracy":
            metric_output = nlp_compute_accuracy(
                self.model, dataloader, device=self.device, fp16=False
            )
        elif metric == "nlp_perplexity":
            metric_output = nlp_compute_mean_loss(
                self.model, dataloader, device=self.device, fp16=False
            )
        elif metric == "cv_accuracy":
            metric_output = cv_compute_accuracy(
                self.model, dataloader, device=self.device, fp16=False
            )
        else:
            raise NotImplementedError(f"{metric} is not supported")

        return metric_output
