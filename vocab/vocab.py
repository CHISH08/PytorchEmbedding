import torch
class Vocab(torch.nn.Module):
    '''
    Словарь эмбеддингов
    '''
    def __init__(self, word, embedding_size, device='cpu'):
        super().__init__()
        uniq_word = list(set(word))
        self.vocab = {uniq_word[i]: i for i in range(len(uniq_word))}
        self.size = len(self.vocab)
        self.W = torch.nn.Embedding(self.size, embedding_size, device=device)

    def __len__(self):
        return self.size

    def forward(self, word_idx_list):
        '''
        Получение эмбеддингов по заданному списку слов
        '''
        return self.W(word_idx_list)

    def text_to_idx(self, word):
        '''
        Превращает строки текста в индексы
        '''
        return torch.tensor([self.vocab[elem] for elem in word], dtype=torch.int64)

    def __getitem__(self, word):
        word_idx = self.vocab.get(word)
        return word_idx