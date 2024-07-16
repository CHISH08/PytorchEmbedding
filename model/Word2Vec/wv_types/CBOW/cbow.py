from torch.utils.data import Dataset
import torch

class CBOWDataset(Dataset):
    def __init__(self, tokens, ws):
        super().__init__()
        self.tokens = torch.tensor(tokens)
        self.ws = ws

    def __len__(self):
        return len(self.tokens) - 2*self.ws

    def __getitem__(self, index):
        start = index
        mid = index + self.ws
        end = index + 2 * self.ws + 1

        context = torch.cat((self.tokens[start:mid], self.tokens[mid+1:end]))
        target = self.tokens[mid]

        return context, target.unsqueeze(-1)
