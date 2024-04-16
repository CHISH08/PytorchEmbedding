import torch
from torch.utils.data import DataLoader, Dataset
from vocab import Vocab
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

class GloVeData(Dataset):
    def __init__(self, co_path, vocab_size):
        super().__init__()
        self.co_path = co_path
        self.vocab_size = vocab_size
        self.cache = {}

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, index):
        if self.cache.get(index):
            co_tensor = torch.tensor(self.cache.get(index), dtype=torch.int32)
        else:
            f = open(self.co_path + '/' + str(index), 'r')
            text = f.read()
            f.close()
            lst = list(map(int, text.split()))
            self.cache[index] = lst
            co_tensor = torch.tensor(lst, dtype=torch.int32)
        return index, co_tensor

def find_indices(lst):
    indices = dict()
    lst = lst.tolist()
    for index, element in enumerate(lst):
        indices.setdefault(element, []).append(index)
    return indices

class GloVe(torch.nn.Module):
    def __init__(self, tokens, embedding_size, window_size, num_workers, device='cuda'):
        super().__init__()
        self.num_workers = num_workers
        self.embed = Vocab(tokens, embedding_size)
        self.vocab_size = len(self.embed)
        self.linear = torch.nn.Linear(embedding_size, self.vocab_size, device=device, dtype=torch.float32)
        self.b1 = torch.randn((self.vocab_size, 1), dtype=torch.float32, device=device, requires_grad=True)
        self.b2 = torch.randn((1, self.vocab_size), dtype=torch.float32, device=device, requires_grad=True)

        self.optim = torch.optim.Adam(list(self.embed.parameters()) + list(self.linear.parameters()) + [self.b1, self.b2], lr=3e-4)

        self.X_max = self.build_matrix(self.embed.text_to_idx(tokens), "./GloVe/glove_co", self.vocab_size, window_size)
        self.device = device

    def train(self, num_epochs, batch_size):
        dataset = GloVeData('./GloVe/glove_co', self.vocab_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers)
        for epoch in range(num_epochs):
            self.optim.zero_grad()
            total_loss = 0
            for batch_idx, Co_matrix in dataloader:
                J = self.glove_loss(self(batch_idx), Co_matrix, self.X_max, self.device)
                dataset.cache.clear()
                J.backward()
                self.optim.step()
                total_loss += J.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

    def forward(self, batch_idx):
        embeddings1 = self.embed(batch_idx).to(self.device)
        return self.linear(embeddings1) + self.b1[batch_idx] + self.b2

    def euclid_dist(self, word, k):
        word = self.embed.vocab.get(word)
        if not word:
            return None

        full_range = torch.arange(len(self.embed.vocab))
        square_distances = torch.sum(torch.pow(self.embed(full_range) - self.embed(torch.tensor(word)), 2), dim = -1)
        k_min_euclid = torch.argsort(square_distances)[1:k+1]
        word_list = list(self.embed.vocab.keys())
        k_min_euclid_word = list(map(lambda idx: word_list[idx], k_min_euclid.tolist()))
        return k_min_euclid_word, self.embed(k_min_euclid)

    def __getitem__(self, text):
        return self.embed[text]

    @staticmethod
    def glove_loss(X, matrix, X_max, device):
        def weight_param(X, X_max):
            return torch.pow(X/X_max, 3/4)
        weight = weight_param(matrix, X_max).to(device)
        J = torch.sum(weight * torch.pow(X - torch.log1p(matrix.to(device)), 2))
        return J

    @staticmethod
    def process_word(text, word, word_d, vocab_size, window_size, path_file):
        Co_matrix = [0 for _ in range(vocab_size)]
        for i in word_d:
            for j in range(1, window_size + 1):
                if i - j >= 0:
                    Co_matrix[text[i - j]] += 1
                if i + j < vocab_size:
                    Co_matrix[text[i + j]] += 1

        with open(os.path.join(path_file, str(word)), 'w') as f:
            f.write(' '.join(map(str, Co_matrix)))

        return max(Co_matrix)

    def build_matrix(self, text, path_file, vocab_size, window_size):
        if os.path.exists(path_file) and os.path.isdir(path_file):
            shutil.rmtree(path_file)
        os.makedirs(path_file, exist_ok=True)

        word_d = find_indices(text)

        X_max_values = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_word, text, word, word_d[word], vocab_size, window_size, path_file) for word in word_d]
            for future in as_completed(futures):
                X_max_values.append(future.result())

        return max(X_max_values)
