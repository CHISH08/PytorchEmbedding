from ..Word2Vec import Word2Vec
class FastText(Word2Vec):
    def __getitem__(self, word_idx):
        return super().__getitem__(word_idx).mean(dim=0)

    def k_Nearest(self, words_n_grams, *args, **kwargs):
        return super().k_Nearest(*args, words_n_grams=words_n_grams, **kwargs)
