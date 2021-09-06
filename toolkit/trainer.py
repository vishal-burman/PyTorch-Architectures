from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import get_linear_schedule_with_warmup


def init_optimizer(optimizer: str, model, lr):
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
        model,
        train_dataset,
        valid_dataset,
        fp16=False,
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
        optimizer="adam",
        scheduler="linear",
        lr=3e-5,
        epochs=3,
        batch_size=32,
        shuffle_train=False,
        shuffe_valid=False,
        num_warmup_steps=0,
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
            if not self.model.training:
                print("Model is in eval mode ... switching to train mode")
                model.train()

            for idx, sample in train_loader:
                outputs = self.model(**dict_to_device(sample, device=self.device))
                loss, _ = outputs[0]
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

    def validate(
        self,
        dataloader,
        metric,
    ):
        if self.model.training:
            print("Model is in train mode ... switching to eval mode")
            self.model.eval()

        if metric == "nlp_accuracy":
            pass
        elif metric == "nlp_perplexity":
            pass
        elif metric == "cv_accuracy":
            pass
        else:
            raise NotImplementedError(f"{metric} is not supported")
