import torch
from bitarray import bitarray
from math import ceil, log2
class HierarchicalSoftmax:
    def __init__(self, vocab):
        self.vocab = vocab
        self.build_tree(list(vocab.keys()))

    def build_tree(self, words):
        self.n = ceil(log2(len(words))) if words else 0
        self.huffman_word = {}
        self.huffman_leaf = {}
        stack = [(words, self.n, bitarray())]

        while stack:
            words, n, hp = stack.pop()
            n -= 1
            if words:
                if n == -1:
                    self.huffman_word[words[0]] = hp
                else:
                    self.huffman_leaf[hp.to01()] = len(self.huffman_leaf)
                    stack.append((words[:2 ** n], n, hp + bitarray('0')))
                    stack.append((words[2 ** n:], n, hp + bitarray('1')))

    def __getitem__(self, word):
        huffman_word = self.huffman_word[word]
        huffman_word_string = huffman_word.to01()
        road = torch.tensor([self.huffman_leaf[huffman_word_string[:i]] for i in range(len(huffman_word_string))])
        return road, torch.tensor(huffman_word)
