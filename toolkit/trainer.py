from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .utils import get_linear_schedule_with_warmup, dict_to_device
from .metrics import cv_compute_accuracy, nlp_compute_accuracy, nlp_compute_mean_loss


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
        train_dataset: Dataset,
        valid_dataset: Dataset,
        fp16: bool = False,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if fp16:
            assert (
                torch.cuda.is_available()
            ), f"fp16 available only for CUDA devices, found {self.device}"
            raise NotImplementedError

    def train(
        self,
        optimizer: str = "adam",
        scheduler: str = "linear",
        lr: float = 3e-5,
        epochs: int = 3,
        batch_size: int = 32,
        shuffle_train: bool = False,
        shuffe_valid: bool = False,
        num_warmup_steps: int = 0,
        metric: str = "nlp_perplexity",
    ):
        if not self.model.training:
            print("Model in eval mode...switching to train mode")
            self.model.train()

        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle_train
        )
        valid_loader = DataLoader(
            self.valid_dataset, batch_size=batch_size, shuffle=shuffle_valid
        )

        num_training_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        optimizer = init_optimizer(optimizer, self.model, lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=-1,
        )

        for epoch in range(epochs):
            loss_list = []
            if not self.model.training:
                print("Model is in eval mode ... switching to train mode")
                self.model.train()

            for idx, sample in train_loader:
                outputs = self.model(**dict_to_device(sample, device=self.device))
                loss, _ = outputs[0]
                loss_list.append(loss.item())
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            self.model.eval()
            with torch.set_grad_enabled(False):
                metric_output = self.validate(valid_loader, metric=metric)

            mean_loss = torch.mean(torch.tensor(loss_list)).item()
            print(
                f"Epoch: {epoch + 1} || Training Loss: {mean_loss:.3f} || {metric}: {metric_output:.3f}"
            )

    def validate(
        self,
        dataloader,
        metric,
    ):
        if self.model.training:
            print("Model is in train mode ... switching to eval mode")
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
