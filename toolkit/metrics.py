import math
import warnings
import torch
import torch.nn.functional as F
from .utils import dict_to_device


def cv_compute_accuracy(model, data_loader, device, fp16=False):
    if not torch.cuda.is_available() and fp16:
        raise RuntimeError("fp16 only available for cuda devices")

    if model.training:
        warnings.warn("Model is in training mode...switching to eval mode")
        model.eval()

    correct, total = 0, 0
    with torch.set_grad_enabled(False):
        for sample in data_loader:
            if "pixel_values" in sample and "labels" in sample:
                assert (
                    sample["pixel_values"].dim() == 4
                ), "Images should be 4-dimensional"
                assert sample["pixel_values"].size(0) == sample["labels"].size(
                    0
                ), "Number of Images and Labels should be same"
            elif type(sample) == list and (
                isinstance(sample[0], torch.Tensor)
                and isinstance(sample[1], torch.Tensor)
            ):
                assert sample[0].dim() == 4, "Images should be 4-dimensional"
                assert sample[0].size(0) == sample[1].size(
                    0
                ), "Number of Images and Labels should be same"
                sample = {"pixel_values": sample[0], "labels": sample[1]}
            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**dict_to_device(sample, device))
            else:
                outputs = model(**dict_to_device(sample, device))
            if type(outputs) == tuple:
                loss = outputs[0]
                logits = outputs[1]
            else:
                logits = outputs

            labels = sample["labels"].to(device)
            prob = F.softmax(logits, dim=-1)
            _, preds = torch.max(prob, dim=1)
            correct += (preds == labels).sum()
            total += labels.size(0)
    return (correct.float() / total * 100).item()


def nlp_compute_accuracy(model, data_loader, device, fp16=False):
    if not torch.cuda.is_available() and fp16:
        raise RuntimeError("fp16 available only for cuda devices")

    if model.training:
        warnings.warn("Model is in training mode...switching to eval mode")
        model.eval()

    correct, total = 0, 0
    with torch.set_grad_enabled(False):
        for sample in data_loader:
            if sample["input_ids"].dim() == 3:
                assert (
                    sample["input_ids"].shape == sample["attention_mask"].shape
                ), "input_ids and attention_mask shape do not match"
                input_ids = sample["input_ids"].squeeze(1).to(device)
                attention_mask = sample["attention_mask"].squeeze(1).to(device)
                labels = sample["labels"].to(device)
            else:
                assert (
                    sample["input_ids"].shape == sample["attention_mask"].shape
                ), "input_ids and attention_mask shape do not match"
                input_ids = sample["input_ids"].to(device)
                attention_mask = sample["attention_mask"].to(device)
                labels = sample["labels"].to(device)

            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if type(outputs) is tuple:
                loss = outputs[0]
                logits = outputs[1]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
                loss = None
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
            else:
                logits = outputs

            prob = F.softmax(logits, dim=-1)
            _, preds = torch.max(prob, dim=1)
            correct += (preds == labels).sum()
            total += labels.size(0)
    return (correct.float() / total * 100).item()


def nlp_compute_mean_loss(model, data_loader, device, fp16=False):
    if not torch.cuda.is_available() and fp16:
        raise RuntimeError("fp16 available only for cuda devices")

    if model.training:
        warnings.warn("Model is in training mode...switching to eval mode")
        model.eval()
    loss_list = []
    with torch.set_grad_enabled(False):
        for sample in data_loader:
            if sample["input_ids"].dim() == 3:
                assert (
                    sample["labels"].dim() == 3
                ), "Shape of input_ids and labels do not match"
                input_ids = sample["input_ids"].squeeze(1).to(device)
                attention_mask = sample["attention_mask"].squeeze(1).to(device)
                labels = sample["labels"].squeeze(1).to(device)
            else:
                assert sample["labels"].dim() == 2
                input_ids = sample["input_ids"].to(device)
                attention_mask = sample["attention_mask"].to(device)
                labels = sample["labels"].to(device)

            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
            if type(outputs) is tuple:
                loss = outputs[0].item()
            elif hasattr(outputs, "loss"):
                loss = outputs.loss.item()
            else:
                loss = outputs.item()
            loss_list.append(loss)
    return math.exp(torch.tensor(loss_list).mean().item())
