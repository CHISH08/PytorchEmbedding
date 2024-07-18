from torch import nn
from .encoder import Encoder
from .decoder import Decoder

class GPT(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, num_layers, vocab_size):
        super().__init__()
        self.decoder = Decoder(embed_dim, hidden_size, num_heads, num_layers)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        hidden = self.decoder(x, x, memory_key_padding_mask=mask)
        output = self.linear(hidden)
        return output

    def predict(self, x, return_hidden, mask=None):
        hidden = self.decoder(x, x, memory_key_padding_mask=mask)
        output = self.linear(hidden)
        if return_hidden:
            output = {'pred_idx': output, 'hidden': hidden}
        return output

class BERT(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, num_layers):
        super().__init__()
        self.encoder = Encoder(embed_dim, hidden_size, num_heads, num_layers)

    def forward(self, x, mask=None):
        output = self.encoder(x, src_attention_mask=mask)
        return output

class T5(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, num_layers, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = Encoder(embed_dim, hidden_size, num_heads, num_layers)
        self.decoder = Decoder(embed_dim, hidden_size, num_heads, num_layers)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt, src_attention_mask=None, memory_key_padding_mask=None):
        src = self.embedding(src)
        encoder_output = self.encoder(src, src_attention_mask)

        tgt = self.embedding(tgt)
        output = self.decoder(tgt, encoder_output, memory_key_padding_mask)

        output = self.linear(output)

        return output
