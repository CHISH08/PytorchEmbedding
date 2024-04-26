from torch.utils.data import Dataset

class SGDataset(Dataset):
    def __init__(self, tokens, ws):
        super().__init__()
        self.tokens = tokens
        self.ws = ws
        self.total_pairs = (len(self.tokens) - 2 * self.ws) * (2 * ws)

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, idx):
        sample_index = idx // (2 * self.ws)
        offset = idx % (2 * self.ws)

        if offset >= self.ws:
            offset += 1

        start = sample_index
        mid = sample_index + self.ws
        end = sample_index + 2 * self.ws + 1

        context = self.tokens[mid]
        target = self.tokens[start + offset]

        return context.unsqueeze(0), target