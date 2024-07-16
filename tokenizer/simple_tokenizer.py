from nltk.tokenize import WordPunctTokenizer
import pickle
class AutoTokenizer:
    def __init__(self, pretrained=None):
        self._tokenizer = WordPunctTokenizer()
        self.special_tokens = {
            '<UNK>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            '<BOP>': 3,
            '<EOP>': 4,
            '<BOT>': 5,
            '<EOT>': 6
        }
        self.vocab, self.words = self._from_pretrained(pretrained) if pretrained else (self.special_tokens.copy(), list(self.special_tokens.keys()))

    def train(self, text: str):
        tokens = self._tokenize(text)
        unique_tokens = set(tokens)
        add_tokens = list(unique_tokens - set(self.words))
        add_vocab = {add_token: i + len(self.vocab) for i, add_token in enumerate(add_tokens)}
        self.vocab = self.vocab | add_vocab
        self.words.extend(add_tokens)

    def encode(self, text: str, add_special_tokens=True):
        tokens = self._tokenize(text)
        tokens_idx = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return tokens_idx

    def decode(self, token_idxs: list[int], join=False):
        tokens = [self.words[idx] for idx in token_idxs]
        text = self._tokens_join(tokens, join)
        return text

    def save_pretrained(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self.vocab, f)

    def _tokenize(self, text: str):
        tokens = self._tokenizer.tokenize(text)
        return tokens

    @staticmethod
    def _tokens_join(text: list[str], join):
        if join:
            text = ' '.join(text)
            text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
        return text

    @staticmethod
    def _from_pretrained(file_path: str):
        with open(file_path, 'rb') as f:
            vocab = pickle.load(f)
        words = list(vocab.keys())
        return vocab, words
