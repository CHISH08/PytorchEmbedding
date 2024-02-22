import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import re
from vocab import Vocab
import pymorphy2
from nltk.corpus import stopwords
import time

class TextDataset(Dataset):
    def __init__(self, text, window_size):
        self.text = text
        self.window_size = window_size

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        start = idx - self.window_size
        end = idx + self.window_size + 1
        full_range = torch.arange(start, end)
        negative_left_elements = (full_range < 0)
        negative_right_elements = (full_range >= len(self.text))
        left_negative = torch.any(negative_left_elements)
        right_negative = torch.any(negative_right_elements)
        if left_negative and right_negative:
            raise Exception("Слишком большой windows_size или маленький размер текста! Измените параметры модели...")
        elif left_negative:
            full_range[negative_left_elements] = -full_range[negative_left_elements] + 2*idx
        elif right_negative:
            full_range[negative_right_elements] -= (end - start - 1)
        context = self.text[full_range[full_range != idx]]
        target = self.text[idx]

        return context, target

class CBOW(nn.Module):
    '''
    Реализация модели CBOW
    '''
    def __init__(self, text, embedding_size=100, windows_size=1, lr=0.005, num_epochs=1, batch_size=2000, device='cpu', num_workers=1, log=1, pretrained=None, tfidf=False):
        super().__init__()
        self.device = device
        self.vocab = Vocab(text, embedding_size, pretrained, windows_size, device)
        self.linear = nn.Linear(embedding_size, len(self.vocab), device=self.device)
        self.activation = nn.LogSoftmax(dim=1)
        self.windows_size = windows_size
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.num_workers = num_workers
        self.train(num_epochs, batch_size, log)

    def forward(self, word_list):
        '''
        По списку слов получить матрицу вероятности
        '''
        idx_embed = self.vocab(word_list)
        embeds = torch.sum(idx_embed, dim=1)
        out = self.linear(embeds)
        probs = self.activation(out)
        return probs

    def train(self, num_epochs, batch_size, log):
        '''
        Обучение модели
        '''
        dataset = TextDataset(self.vocab.text, self.windows_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, shuffle=True)
        first_time = time.time()
        loss_list = []
        accuracy_list = []

        for epoch in range(1, num_epochs+1):
            total_loss = 0
            total_accuracy = 0

            for context, target in dataloader:
                context, target = context.to(self.device), target.to(self.device)
                self.optim.zero_grad()
                probs = self.forward(context)
                loss = self.loss(probs, target)
                loss.backward()
                self.optim.step()

                total_loss += loss.item() * target.size(0) / len(dataset)
                _, predicted = torch.max(probs, 1)
                total_accuracy += (predicted == target).sum().item() / len(dataset) * 100

            if log == 1:
                print(f'Epoch [{epoch}/{num_epochs}, time: {(time.time() - first_time)/60:.3f} minutes]: Loss {total_loss:.6f}, Accuracy {total_accuracy:.3f}%')
            elif log != 0 and log != 1:
                loss_list.append(total_loss)
                accuracy_list.append(total_accuracy)
        if log != 0 and log != 1:
            self.savefig(loss_list, accuracy_list, time.time() - first_time, log)

    def __getitem__(self, word):
        word = word.lower()
        idx = torch.tensor(self.vocab[word])
        return self.vocab.W(idx.to(self.device)).detach().cpu()

    def euclid_dist(self, word, k):
        word = word.lower()
        word = self.vocab.vocab.get(word)
        full_range = torch.arange(len(self.vocab.vocab)).to(self.device)
        square_distances = torch.sum(torch.pow(self.vocab.W(full_range) - self.vocab.W(torch.tensor(word).to(self.device)), 2), dim = -1)
        k_min_euclid = torch.argsort(square_distances)[1:k+1]
        word_list = list(self.vocab.vocab.keys())
        k_min_euclid_word = list(map(lambda idx: word_list[idx], k_min_euclid.detach().cpu().tolist()))
        return k_min_euclid_word, self.vocab.W(k_min_euclid).detach().cpu()

    @staticmethod
    def tokenizer(text):
        '''
        Регулярное выражение для разделения слов и знаков препинания
        '''
        morph = pymorphy2.MorphAnalyzer()
        # pattern = r'\.\.\.|\w+|[^\w\s]'
        pattern = r'\w+'
        text = text.lower()
        tokens = re.findall(pattern, text)
        stop_words = set(stopwords.words('russian'))
        token_list = []

        for i in range(len(tokens)):
            if tokens[i] not in stop_words:
                token_list.append(morph.parse(tokens[i])[0].normal_form)

        return token_list

    @staticmethod
    def savefig(loss, acc, times, path):
        epoch = range(1, len(loss)+1)
        import matplotlib.pyplot as plt
        import os
        if not os.path.isdir(path + "/graphic"):
            os.mkdir(path + "/graphic")
        plt.subplot(211)
        plt.title(f"Время обучения: {times/60:.2f} минут")
        plt.ylabel('CrossEntropyLoss')
        plt.plot(epoch,loss,'b')
        plt.subplot(212)
        plt.xlabel('Эпоха')
        plt.ylabel('Метрика, %')
        plt.plot(epoch,acc,'r')
        plt.tight_layout()
        plt.savefig(path + "/graphic/train.png")
        plt.close()