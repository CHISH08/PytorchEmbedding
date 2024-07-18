import pickle
import re
class CustomTokenizer:
    @staticmethod
    def tokenize(text):
        pattern = re.compile(r'\[[A-Z]+\]|\w+|[^\w\s]')
        return pattern.findall(text)

class AutoTokenizer:
    def __init__(self, pretrained=None):
        self._tokenizer = CustomTokenizer()
        self.special_tokens = {
            '[UNK]': 0,
            '[BOS]': 1,
            '[EOS]': 2,
            '[BOP]': 3,
            '[EOP]': 4,
            '[BOT]': 5,
            '[EOT]': 6
        }
        self.vocab, self.words = self._from_pretrained(pretrained) if pretrained else (self.special_tokens.copy(), list(self.special_tokens.keys()))

    def train(self, text: str):
        tokens = self._tokenize(text)
        unique_tokens = set(tokens)
        add_tokens = list(unique_tokens - set(self.words))
        add_vocab = {add_token: i + len(self.vocab) for i, add_token in enumerate(add_tokens)}
        self.vocab = self.vocab | add_vocab
        self.words.extend(add_tokens)

    def encode(self, text: str, add_special_tokens=False):
        if add_special_tokens:
            text = self._add_special_tokens(text)

        tokens = self._tokenize(text)
        tokens_idx = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        return tokens_idx

    def decode(self, token_idxs: list[int], join=False, add_special_tokens=False):
        tokens = [self.words[idx] for idx in token_idxs]
        if not add_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
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

    @staticmethod
    def _add_special_tokens(text):
        paragraphs = text.split('\n')
        sentence_endings = re.compile(r'(?<=[.!?])\s+|\.\.\.')

        tokenized_paragraphs = []
        for paragraph in paragraphs:
            paragraph_sentences = sentence_endings.split(paragraph)
            paragraph_sentences = [f'[BOS] {sentence.strip()} [EOS]' for sentence in paragraph_sentences if sentence.strip()]
            tokenized_paragraph = ' '.join(paragraph_sentences)
            tokenized_paragraphs.append(f'[BOP] {tokenized_paragraph} [EOP]')
        
        processed_text = '\n'.join(tokenized_paragraphs)
        return processed_text
    