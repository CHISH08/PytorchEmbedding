from torch import nn
import torch
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class DynamicalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mult = -math.log(10000.0) / self.embed_dim

    def forward(self, x):
        seq_length = x.size(0)
        pe = torch.zeros(seq_length, self.embed_dim).to(x.device)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1).to(x.device)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float) * self.mult).to(x.device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)
        x = x + pe[:x.size(0), :]
        return x
