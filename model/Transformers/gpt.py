import math
import torch
from torch import nn
from torch.utils.data import Dataset
from .model import GPT as GPTBody
from .base import TextGenerator

class TextDataset(Dataset):
    def __init__(self, tokens, seq_size):
        self.tokens = tokens
        self.seq_size = seq_size

    def __getitem__(self, idx):
        context = self.tokens[idx:idx+self.seq_size]
        target = self.tokens[idx+1:idx+1+self.seq_size]
        return torch.tensor(context), torch.tensor(target)

    def __len__(self):
        return len(self.tokens) - self.seq_size - 1

class GPT(GPTBody, TextGenerator):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_heads, num_layers, seq_size, lr=3e-4, test_size=0.8, device='cpu'):
        GPTBody.__init__(self, embed_dim, hidden_size, num_heads, num_layers, vocab_size)
        self.device = device
        self.test_size = test_size
        self.ws = seq_size
        self.vocab_size = vocab_size

        self.W1 = nn.Embedding(vocab_size, embed_dim)
        self.W1_weight = self.W1.weight
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.metric = lambda x: math.exp(x)
        self.dataset = TextDataset

        self.to(self.device)

    def forward(self, idxs):
        self.embed = self.W1(idxs)
        output = GPTBody.forward(self, self.embed).view(-1, self.vocab_size)
        return output

    def predict(self, idxs, return_hidden=False):
        self.embed = self.W1_weight[idxs]
        output = GPTBody.predict(self, self.embed, return_hidden)
        return output
