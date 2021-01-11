import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super().__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class BPR(nn.Module):
    def __init__(self, n_users, n_items, embedding_size):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()
        
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                xavier_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                xavier_normal_(m.weight.data)
                if m.bias is not None:
                    constant_(m.bias.data, 0)

    def get_user_embedding(self, user):
        """ Returns batch of user embedding based on input user's id """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        """ Returns batch of item embedding based on input item's id """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_emb = self.get_user_embedding(user)
        item_emb = self.get_item_embedding(item)
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        user = interaction['user_id']
        pos_item = interaction['pos_item_id']
        neg_item = interaction['neg_item_id']
        user_emb, pos_emb = self.forward(user, pos_item)
        neg_emb = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_emb, pos_emb).sum(dim=1), torch.mul(user_emb, neg_emb).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user = interaction['user_id']
        user_emb = self.get_user_embedding(user)
        all_item_emb = self.item_embedding.weight
        score = torch.matmul(user_emb, all_item_emb.transpose(0, 1))
        return score.view(-1)

