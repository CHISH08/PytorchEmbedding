import torch
import math
import torch.nn as nn
from ..body import BodyModel
from .wv_types import CBOWDataset
from .wv_types import SGDataset

class Word2Vec(BodyModel):
    def __init__(self, dim: int, ws: int, vocab_size: int, sg: bool=False, lr: float=3e-4, test_size: float=0.2, device: str='cpu'):
        super().__init__()
        self.device = device
        self.ws = ws
        self.sg = sg
        self.test_size = test_size

        self.W1 = nn.Embedding(vocab_size, dim)
        self.W1_weight = self.W1.weight
        self.W2 = nn.Embedding(vocab_size, dim)
        self.W2_weight = self.W2.weight

        if sg:
            self.dataset = SGDataset
        else:
            self.dataset = CBOWDataset

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.metric = lambda x: math.exp(x)

        self.to(device)

    def __getitem__(self, word_idx):
        self.eval()
        return self.W1_weight[word_idx].detach().cpu().clone()

    def forward(self, X):
        X = self.W1(X)
        X = torch.mean(X, dim=-2)
        X = torch.mm(X, self.W2_weight.T)
        return X
