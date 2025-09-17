import torch.nn as nn


class GMFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.affine_transform = nn.Linear(embedding_dim, 1)
        self.user_embeddings.weight.data.uniform_(0.5, 1.0)
        self.item_embeddings.weight.data.uniform_(0.5, 1.0)

    def forward(self, users, items):
        user_emb = self.user_embeddings(users)
        item_emb = self.item_embeddings(items)
        return self.affine_transform(user_emb * item_emb).squ
