from torch import nn
import torch

class AttentionLayer(nn.Module):
    def __init__(self, word_size, keys_size, values_size):
        super().__init__()
        self.Wq = nn.Linear(word_size, keys_size)
        self.Wk = nn.Linear(word_size, keys_size)
        self.Wv = nn.Linear(word_size, values_size)
        self.keys_size = keys_size
    
    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        S = torch.matmul(Q, K.transpose(1, 2)) / (self.keys_size**0.5)
        A = torch.softmax(S, dim=-1)
        Z = torch.matmul(A, V)
        return Z

class MultiHeadAttention(nn.Module):
    def __init__(self, word_size, num_heads):
        super().__init__()
        assert word_size % num_heads == 0, "word_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = word_size // num_heads

        self.Wq = nn.Linear(word_size, word_size)
        self.Wk = nn.Linear(word_size, word_size)
        self.Wv = nn.Linear(word_size, word_size)
        self.Wo = nn.Linear(word_size, word_size)

    def forward(self, x, y):
        batch_size = x.size(0)
        seq_length = x.size(1)

        Q = self.Wq(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = self.Wk(y).view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = self.Wv(y).view(batch_size, seq_length, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        S = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        A = torch.softmax(S, dim=-1)

        O = torch.matmul(A, V)
        O = O.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        O = self.Wo(O)

        return O

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, word_size, num_heads):
        super().__init__()
        assert word_size % num_heads == 0, "word_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = word_size // num_heads

        self.Wq = nn.Linear(word_size, word_size)
        self.Wk = nn.Linear(word_size, word_size)
        self.Wv = nn.Linear(word_size, word_size)
        self.Wo = nn.Linear(word_size, word_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        mask = torch.tril(torch.ones((seq_length, seq_length), device=x.device)).unsqueeze(0).unsqueeze(0)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        Q = self.Wq(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = self.Wk(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = self.Wv(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        S = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        S = S + mask
        A = torch.softmax(S, dim=-1)

        O = torch.matmul(A, V)
        O = O.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        O = self.Wo(O)

        return O
