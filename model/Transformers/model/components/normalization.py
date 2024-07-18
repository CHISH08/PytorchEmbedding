from torch import nn
import torch
class LayerNormalization(nn.Module):
    def __init__(self, embed_dim, epsilon=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.epsilon = epsilon

    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x_normalized = (x - mu) / (std + self.epsilon)
        y = self.gamma * x_normalized + self.beta
        return y
