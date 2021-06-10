import math
import warnings
import torch
import torch.nn.functional as F

def cv_compute_accuracy(model, data_loader, device):
    if model.training:
        warnings.warn('Model is in training mode...switching to eval mode')
        model.eval()
    correct, total = 0, 0
    with torch.set_grad_enabled(False):
        for sample in data_loader:
            if 'img' in sample and 'labels' in sample:
                assert sample['img'].dim() == 4, 'Images should be 4-dimensional'
                assert sample['img'].size(0) == sample['labels'].size(0), 'Number of Images and Labels should be same'
                img = sample['img'].to(device)
                labels = sample['labels'].to(device)
            elif type(sample) == list and (isinstance(sample[0], torch.Tensor) and isinstance(sample[1], torch.Tensor)):
                assert sample[0].dim() == 4, 'Images should be 4-dimensional'
                assert sample[0].size(0) == sample[1].size(0), 'Number of Images and Labels should be same'
                img = sample[0].to(device)
                labels = sample[1].to(device)
            outputs = model(img, labels=labels)
            if type(outputs) == tuple:
                loss = outputs[0]
                logits = outputs[1]
            else:
                logits = outputs

            prob = F.softmax(logits, dim=-1)
            _, preds = torch.max(prob, dim=1)
            correct += (preds == labels).sum()
            total += labels.size(0)
    return (correct.float() / total * 100).item()


def nlp_compute_accuracy(model, data_loader, device):
    if model.training:
        warnings.warn('Model is in training mode...switching to eval mode')
        model.eval()
    correct, total = 0, 0
    with torch.set_grad_enabled(False):
        for sample in data_loader:
            if sample['input_ids'].dim() == 3:
                assert sample['input_ids'].shape == sample['attention_mask'].shape, 'input_ids and attention_mask shape do not match'
                input_ids = sample['input_ids'].squeeze(1).to(device)
                attention_mask = sample['attention_mask'].squeeze(1).to(device)
                labels = sample['labels'].to(device)
            else:
                assert sample['input_ids'].shape == sample['attention_mask'].shape, 'input_ids and attention_mask shape do not match'
                input_ids = sample['input_ids'].to(device)
                attention_mask = sample['attention_mask'].to(device)
                labels = sample['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if type(outputs) is tuple:
                loss = outputs[0]
                logits = outputs[1]
            elif hasattr(outputs, 'logits'): 
                logits = outputs.logits
                loss = None
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
            else:
                logits = outputs

            prob = F.softmax(logits, dim=-1)
            _, preds = torch.max(prob, dim=1)
            correct += (preds == labels).sum()
            total += labels.size(0)
    return (correct.float() / total * 100).item()

def nlp_compute_mean_loss(model, data_loader, device):
    if model.training:
        warnings.warn('Model is in training mode...switching to eval mode')
        model.eval()
    loss_list = []
    with torch.set_grad_enabled(False):
        for sample in data_loader:
            if sample['input_ids'].dim() == 3:
                assert sample['labels'].dim() == 3, 'Shape of input_ids and labels do not match'
                input_ids = sample['input_ids'].squeeze(1).to(device)
                attention_mask = sample['attention_mask'].squeeze(1).to(device)
                labels = sample['labels'].squeeze(1).to(device)
            else:
                assert sample['labels'].dim() == 2
                input_ids = sample['input_ids']
                attention_mask = sample['attention_mask']
                labels = sample['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if type(outputs) is tuple:
                loss = outputs[1].item()
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss.item()
            else:
                loss = outputs.item()
            loss_list.append(loss)
    return math.exp(torch.tensor(loss_list).mean().item())
