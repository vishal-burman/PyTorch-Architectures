import torch
import torch.nn as nn
import torch.nn.functional as F

class BPR(nn.Module):
    def __init__(self,):
        super().__init__()

    def get_user_embedding(self, user):
        pass

    def get_item_embedding(self, item):
        pass

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        pass

    def predict(self, interaction):
        pass

