def build_vocab(words):
    vocab = {}
    for elem in words:
        vocab.setdefault(elem, len(vocab))
    return vocab
