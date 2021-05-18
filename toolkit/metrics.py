import warnings
import torch
import torch.nn.functional as F

def compute_accuracy(model, data_loader, device):
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if type(outputs) is tuple:
                logits = outputs[0]
                loss = outputs[1]
            else:
                logits = outputs

            prob = F.softmax(logits, dim=-1)
            _, preds = torch.max(prob, dim=1)
            correct += (preds == labels).sum()
            total += labels.size(0)
    return correct.float() / total * 100
