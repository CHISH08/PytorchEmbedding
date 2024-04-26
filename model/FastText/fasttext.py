from Word2Vec import Word2Vec
from Vocab import build_vocab
import torch

def fast_tokenizer(tokens, n_gram):
    tokens_gram = []
    for token in tokens:
        token = f"<{token}>"
        tokens_gram.extend(token[i:i+n_gram] for i in range(len(token) - n_gram + 1))
    return tokens_gram

class FastText:
    def __init__(self, tokens, dim, ws, n_gram, sg=0, device='cpu'):
        self.vocab = build_vocab(tokens)
        fast_tokens = fast_tokenizer(tokens, n_gram)
        self.model = Word2Vec(fast_tokens, dim, ws, sg=sg, device=device)
        self.n_gram = n_gram
        self.dim = dim

    def __getattr__(self, name):
        """
        Если метод не найден в этом классе, ищем его в self.model.
        """
        attr = getattr(self.model, name)

        if callable(attr):
            def method(*args, **kwargs):
                result = attr(*args, **kwargs)
                if name == "fit":
                    self.all_vecs = torch.stack([self[idx] for idx in self.vocab], dim=0)
                return result
            return method
        else:
            return attr

    def __getitem__(self, word):
        word = f"<{word}>"
        n_grams = [word[i:i + self.n_gram] for i in range(len(word) - self.n_gram + 1)]
        word_vector = torch.zeros(self.dim)
        count = 0

        for n_gram in n_grams:
            if n_gram in self.model.vocab:
                word_vector += self.model[n_gram]
                count += 1
        if count > 0:
            word_vector /= count

        return word_vector

    def k_Nearest(self, word, k, return_embed=False, use_cosine=False):
        word_vec = self[word]

        if use_cosine:
            word_vec = word_vec / word_vec.norm(dim=-1, keepdim=True)
            alls_vecs = self.all_vecs / self.all_vecs.norm(dim=-1, keepdim=True)

            if len(word_vec.shape) == 1:
                word_vec = word_vec.unsqueeze(0)

            cosine_similarity = torch.mm(alls_vecs, word_vec.t()).squeeze()
            distances = 1 - cosine_similarity
        else:
            distances = torch.sqrt(torch.sum(torch.pow(self.all_vecs - word_vec, 2), dim=-1))

        k_min_dist = torch.argsort(distances)[:k+1]
        word_list = list(self.vocab)
        k_min_dist_word = [word_list[idx] for idx in k_min_dist.detach().cpu().tolist()]

        if return_embed:
            embed = torch.tensor([self[idx] for idx in k_min_dist])
            return k_min_dist_word, embed
        
        return k_min_dist_word