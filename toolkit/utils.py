import string
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def dict_to_device(sample_dict, device):
    final_dict = sample_dict.copy()
    if not all(isinstance(x, torch.Tensor) for x in final_dict.values()):
        raise TypeError('Only torch.Tensor values can be shifted to CUDA')
    for key in final_dict:
        value = final_dict[key].to(device)
        final_dict[key] = value
    return final_dict
