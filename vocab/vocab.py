import torch
class Vocab(torch.nn.Module):
    '''
    Словарь эмбеддингов
    '''
    def __init__(self, word, embedding_size, pretrained, windows_size, device):
        super().__init__()
        self.vocab = {}
        # if pretrained:
        #     import pickle
        #     with open(pretrained, 'rb') as file:
        #         self.vocab, W_old = pickle.load(file)

        self.size = len(self.vocab)
        self.text = []
        for wrd in word:
            if wrd not in self.vocab:
                self.vocab[wrd] = self.size
                self.size += 1
            self.text.append(self.vocab[wrd])

        self.text = torch.tensor(self.text, dtype=torch.int64)
        self.W = torch.nn.Embedding(self.size, embedding_size, device=device)

        # if pretrained:
        #     with torch.no_grad():
        #         self.W.weight[:W_old.shape[0], :W_old.shape[1]] = W_old.to(device)

    def __len__(self):
        return self.size

    def forward(self, word_idx_list):
        '''
        Получение эмбеддингов по заданному списку слов
        '''
        return self.W(word_idx_list)

    def __getitem__(self, word):
        return self.vocab[word]

    def save(self, file_name):
        '''
        Сохранить параметры модели
        '''
        import pickle
        with open(file_name, 'rb') as file:
            pickle.dump((self.vocab, self.W), file)