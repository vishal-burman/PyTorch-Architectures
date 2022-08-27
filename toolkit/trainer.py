import logging
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from .metrics import cv_compute_accuracy, nlp_compute_accuracy, nlp_compute_mean_loss
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from .utils import dict_to_device, get_linear_schedule_with_warmup

#logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_grad_flow(named_parameters: Tuple[List[str], List[torch.Tensor]]):
    """
    Plots the gradient flow in each layer with each epoch
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
    """

    layers_name = []
    average_gradients = []
    for name, param in named_parameters:
        if param.requires_grad and "bias" not in name:
            layers_name.append(name)
            average_gradients.append(param.grad.abs().mean())

    plt.plot(average_gradients, alpha=0.3, color="b")
    plt.hlines(0, 0, len(average_gradients) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(average_gradients), 1), layers_name, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(average_gradients))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradients")
    plt.title("Gradient Flow")
    plt.grid(True)


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
        optimizer: str = "adam",
        scheduler: Optional[str] = "linear",
        lr: float = 3e-5,
        epochs: int = 3,
        batch_size: int = 32,
        num_warmup_steps: int = 0,
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        use_amp: bool = False,
        eval_metric: str = "nlp_perplexity",
        show_grad_flow: bool = False,
    ):
        self.model = model

        assert type(train_dataset) == type(
            valid_dataset
        ), f"train_dataset is {type(train_dataset)} and valid_dataset is {type(valid_dataset)}"
        assert type(train_dataset) is Dataset or type(train_dataset) is DataLoader
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        if type(self.train_dataset) is Dataset:
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=shuffle_train
            )
            self.valid_loader = DataLoader(
                self.valid_dataset, batch_size=batch_size, shuffle=shuffle_valid
            )
        else:
            self.train_loader = self.train_dataset
            self.valid_loader = self.valid_dataset

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if use_amp:
            assert (
                torch.cuda.is_available()
            ), f"fp16 available only for CUDA devices, found {self.device}"
            raise NotImplementedError  # TODO

        self.num_training_steps = epochs * len(self.train_loader)
        self.optimizer = init_optimizer(optimizer, self.model, lr)
        if scheduler is not None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.num_training_steps,
                last_epoch=-1,
            )
        else:
            self.scheduler = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_metric = eval_metric
        self.show_grad_flow = show_grad_flow

    def train(
        self,
    ):

        # Details
        logger.info("********** Running Training **********")
        logger.info(f"  Total Training Steps = {self.num_training_steps}  ")
        logger.info(f"  Epochs = {self.epochs}  ")
        logger.info(f"  Batch Size = {self.batch_size}  ")
        logger.info(f"  Length of Train DataLoader = {len(self.train_loader)}  ")
        logger.info(f"  Length of Valid DataLoader = {len(self.valid_loader)}  ")

        progress_bar = tqdm(range(self.num_training_steps))

        for epoch in range(self.epochs):
            loss_list = []
            layers = []
            average_gradients = []
            if not self.model.training:
                self.model.train()

            for idx, sample in enumerate(self.train_loader):
                loss, logits = self.model(**dict_to_device(sample, device=self.device))
                loss_list.append(loss.item())
                loss.backward()
                if self.show_grad_flow:
                    plot_grad_flow(self.model.named_parameters())

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

            self.model.eval()
            with torch.set_grad_enabled(False):
                metric_output = self.validate(
                    self.valid_loader, metric=self.eval_metric
                )

            mean_loss = torch.mean(torch.tensor(loss_list)).item()
            logger.info(
                f"\nEpoch: {epoch + 1} || Training Loss: {mean_loss:.3f} || {self.eval_metric}: {metric_output:.3f}"
            )
            logger.info(f"\nGradient-Flow for epoch {epoch + 1}")
            plt.show()

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
