from torch.utils.data import Dataset
import torch

class SGDataset(Dataset):
    def __init__(self, tokens, ws):
        super().__init__()
        self.tokens = tokens
        self.ws = ws
        self.total_pairs = len(self.tokens) - 2*self.ws

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, index):
        start = index
        mid = index + self.ws
        end = index + 2 * self.ws + 1

        target = torch.cat((self.tokens[start:mid], self.tokens[mid+1:end]))
        context = self.tokens[mid]

        return context.unsqueeze(0), target
