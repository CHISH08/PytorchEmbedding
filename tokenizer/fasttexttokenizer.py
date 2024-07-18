from .simple_tokenizer import AutoTokenizer
class FastTextTokenizer(AutoTokenizer):
    def __init__(self, *args, n_gram=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gram = n_gram

    def train(self, text: str):
        super().train(text)
        self.real_words = list(set(self._tokenizer.tokenize(text)))

    def _tokenize(self, text: str):
        tokens = self._tokenizer.tokenize(text)
        tokens = [sub_token for token in tokens for sub_token in self._word_to_n_grams(token)]
        return tokens

    def _word_to_n_grams(self, word: str):
        if word in self.special_tokens:
            return [word]
        word = '<' + word + '>'
        if len(word) < self.n_gram:
            return [word]
        else:
            token_n_gram = [word[i:i+self.n_gram] for i in range(len(word)-self.n_gram+1)]
            return token_n_gram

    def _tokens_join(self, tokens, join):
        tokens = self._n_gram_to_word(tokens)
        if join:
            tokens = ' '.join(tokens)
            tokens = tokens.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
        return tokens

    def _n_gram_to_word(self, tokens: list[str]):
        untokenized_words = []
        word = ''

        for token in tokens:
            if token in self.special_tokens:
                untokenized_words.append(token)
            elif token[0] == '<' and token[-1] == '>':
                untokenized_words.append(token[1:-1])

            elif token[0] == '<':
                word = token[1:]

            elif token[-1] == '>':
                untokenized_words.append(word)
                word = ''

            else:
                word += token[-1]

        return untokenized_words
