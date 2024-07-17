from torch import nn
from .encoder import Encoder
from .decoder import Decoder

class GPT(nn.Module):
    def __init__(self, word_size, hidden_size, num_heads, num_layers):
        super().__init__()
        self.decoder = Decoder(word_size, hidden_size, num_heads, num_layers)

    def forward(self, x, mask=None):
        output = self.decoder(x, x, memory_key_padding_mask=mask)
        return output

class BERT(nn.Module):
    def __init__(self, word_size, hidden_size, num_heads, num_layers):
        super().__init__()
        self.encoder = Encoder(word_size, hidden_size, num_heads, num_layers)

    def forward(self, x, mask=None):
        output = self.encoder(x, src_attention_mask=mask)
        return output

class T5(nn.Module):
    def __init__(self, word_size, hidden_size, num_heads, num_layers, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_size)
        self.encoder = Encoder(word_size, hidden_size, num_heads, num_layers)
        self.decoder = Decoder(word_size, hidden_size, num_heads, num_layers)
        self.linear = nn.Linear(word_size, vocab_size)

    def forward(self, src, tgt, src_attention_mask=None, memory_key_padding_mask=None):
        src = self.embedding(src)
        encoder_output = self.encoder(src, src_attention_mask)

        tgt = self.embedding(tgt)
        output = self.decoder(tgt, encoder_output, memory_key_padding_mask)

        output = self.linear(output)

        return output
