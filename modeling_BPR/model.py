import torch
import torch.nn as nn
import torch.nn.functional as F

class BPR(nn.Module):
    def __init__(self,):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss() # TODO --> implement loss module
        self.apply(xavier_normal_initialization)

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

