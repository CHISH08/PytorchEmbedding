import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import re
from vocab import Vocab
import time
from updater import HierarchicalSoftmax
from dataset import CBOWDataset, HSDataset
from collections import Counter

class CBOW(nn.Module):
    '''
    Реализация модели CBOW
    '''
    def __init__(self, text, embedding_size=100, windows_size=1, lr=0.005, device='cpu', hs=False, bias=True):
        super().__init__()
        self.device = device
        self.windows_size = windows_size
        self.embedding_size = embedding_size
        self.hs = hs
        self.bias = bias
        self.vocab = Vocab(text, embedding_size, device)
        self.linear = nn.Linear(embedding_size, len(self.vocab), device=self.device, bias=bias)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        if hs:
            self.loss = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            self.loss = nn.CrossEntropyLoss(reduction='sum')

    def __getitem__(self, word):
        word = self.vocab[word]
        idx = torch.tensor(word)
        return self.vocab.W(idx.to(self.device)).detach().cpu()

    def k_Nearest(self, word, k, return_embed=False, use_cosine=False):
        word = self.vocab[word]
        if not word:
            return word
        full_range = torch.arange(len(self.vocab)).to(self.device)
        
        if use_cosine:
            word_vec = self.vocab.W(torch.tensor(word).to(self.device))
            word_vec = word_vec / word_vec.norm(dim=-1, keepdim=True)
            all_vecs = self.vocab.W(full_range)
            all_vecs = all_vecs / all_vecs.norm(dim=-1, keepdim=True)

            # if len(word_vec.shape) == 1:
            #     word_vec = word_vec.unsqueeze(0)

            cosine_similarity = torch.mm(all_vecs, word_vec.t()).squeeze()
            distances = 1 - cosine_similarity
        else:
            square_distances = torch.sum(torch.pow(self.vocab.W(full_range) - self.vocab.W(torch.tensor(word).to(self.device)), 2), dim=-1)
            distances = square_distances
        
        k_min_dist = torch.argsort(distances)[:k+1]
        word_list = list(self.vocab.vocab)
        k_min_dist_word = [word_list[idx] for idx in k_min_dist.detach().cpu().tolist()]
        
        if return_embed:
            embed = self.vocab.W(k_min_dist).detach().cpu()
            return k_min_dist_word, embed
        
        return k_min_dist_word

    def forward2(self, word_list, target_idx):
        '''
        По списку слов получить матрицу вероятности
        '''
        context_embedding = self.vocab(word_list)
        context_embedding_sum = torch.sum(context_embedding, dim=1)
        embed_w_new = self.linear.weight[target_idx]
        embed_w_new_transposed = embed_w_new.transpose(1, 2)
        probs = torch.bmm(context_embedding_sum.unsqueeze(1), embed_w_new_transposed).squeeze(1)
        if self.bias:
            probs = probs + self.linear.bias[target_idx]

        return probs

    def forward1(self, word_list):
        '''
        По списку слов получить матрицу вероятности
        '''
        context_embedding = self.vocab(word_list)
        context_embedding_sum = torch.sum(context_embedding, dim=1)
        probs = self.linear(context_embedding_sum)
        return probs

    def train(self, text, num_epochs, batch_size, num_workers=1, log=1):
        '''
        Обучение модели
        '''
        text = self.vocab.text_to_idx(text)
        dataset = None
        if self.hs:
            self.hs = HierarchicalSoftmax(dict(Counter(text.tolist())))
            self.linear = nn.Linear(self.embedding_size, len(self.hs.huffman_leaf), device=self.device, bias=self.bias)
            dataset = HSDataset(text, self.windows_size, len(self.vocab.vocab), self.hs)
        else:
            dataset = CBOWDataset(text, self.windows_size, len(self.vocab.vocab))
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        first_time = time.time()
        loss_list = []
        accuracy_list = []

        for epoch in range(1, num_epochs+1):
            total_loss = 0

            for data in dataloader:
                context = None
                target = None
                target_idx = None
                if len(data) == 2:
                    context, target = data
                    context, target = context.to(self.device), target.to(self.device)
                elif len(data) == 3:
                    context, target_idx, target = data
                    context, target_idx, target = context.to(self.device), target_idx.to(self.device), target.float().to(self.device)
                self.optim.zero_grad()
                probs = None
                if self.hs:
                    probs = self.forward2(context, target_idx)
                else:
                    probs = self.forward1(context)
                loss = self.loss(probs, target)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

            with torch.no_grad():
                data = next(iter(dataloader))
                context = None
                target = None
                target_idx = None
                if len(data) == 2:
                    context, target = data
                    context, target = context.to(self.device), target.to(self.device)
                elif len(data) == 3:
                    context, target_idx, target = data
                    context, target_idx, target = context.to(self.device), target_idx.to(self.device), target.float().to(self.device)
                probs = None
                total_accuracy = None
                if self.hs:
                    probs = torch.sigmoid(self.forward2(context, target_idx))
                    total_accuracy = ((probs > 0.5).float() == target).sum().item() / self.hs.n / len(target) * 100
                else:
                    probs = self.forward1(context)
                    total_accuracy = (torch.argmax(probs, dim=1) == target).sum().item() / len(target) * 100
                if log == 1:
                    print(f'Epoch [{epoch}/{num_epochs}, time: {(time.time() - first_time)/60:.3f} minutes]: Loss {total_loss:.6f}, Accuracy {total_accuracy:.3f}%')

            loss_list.append(total_loss)
            accuracy_list.append(total_accuracy)

        return loss_list, accuracy_list

    @staticmethod
    def tokenizer(text):
        '''
        Регулярное выражение для разделения слов и знаков препинания
        '''
        pattern = re.compile(r"\w+")
        return pattern.findall(text)

    def save(self, loss, acc, path="."):
        epoch = range(len(loss))
        import os
        if not os.path.isdir(path + "/graphic"):
            os.mkdir(path + "/graphic")
        file_path = path + "/graphic/lossacc.csv"
        if not os.path.exists(path + "/graphic/lossacc.csv"):
            with open(file_path, 'w') as file:
                pass

        with open(file_path, 'a') as file:
            for i in epoch:
                file.write(f"{i};{self.windows_size};{self.embedding_size};{loss[i]};{acc[i]}\n")
    def save_embedding(self, path="."):
        import os
        if not os.path.isdir(path + "/embed"):
            os.mkdir(path + "/embed")
        with open(path + "/embed/w2w.csv", 'w') as file:
            for word in self.vocab.vocab:
                file.write(f"{word};{self[word].tolist()}\n")
