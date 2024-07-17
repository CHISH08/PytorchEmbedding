from ..Word2Vec import Word2Vec
import torch
class GloVe:
    def __init__(self):
        pass

    def forward(self):
        pass

    def fit(self):
        pass

    def build_cooccurrence_matrix(self, tokens):
        words = max(tokens) + 1
        cooccurrence_matrix = torch.zeros((words, words), dtype=torch.int64)
        cooccurrence_matrix = self.update_cooccurrence_matrix(cooccurrence_matrix, self.ws, tokens)
        return cooccurrence_matrix

    @staticmethod
    def update_cooccurrence_matrix(cooccurrence_matrix, ws, tokens):
        tokens_tensor = torch.tensor(tokens, dtype=torch.int64)

        for i in range(len(tokens)):
            start = max(0, i - ws)
            end = min(len(tokens), i + ws + 1)

            context_tokens = torch.cat((tokens_tensor[start:i], tokens_tensor[i+1:end]))
            target_token = tokens_tensor[i]

            cooccurrence_matrix[target_token].index_add_(0, context_tokens, torch.ones_like(context_tokens, dtype=torch.int64))

        return cooccurrence_matrix

    @staticmethod
    def glove_loss():
        pass