from torch import nn
from .components import MultiHeadAttention, LayerNormalization, Add, FeedForward, DynamicalPositionalEncoding

class EncoderBlock(nn.Module):
    def __init__(self, word_size, hidden_size, num_heads):
        super().__init__()
        self.attent = MultiHeadAttention(word_size, num_heads)
        self.norm1 = LayerNormalization(word_size)
        self.norm2 = LayerNormalization(word_size)
        self.add1 = Add(self.norm1)
        self.add2 = Add(self.norm2)
        self.ffn = FeedForward(word_size, hidden_size)

    def forward(self, x, src_attention_mask=None):
        O = self.attent(x, x)
        if src_attention_mask is not None:
            O = O.masked_fill(src_attention_mask == 0, float('-inf'))
        x = self.add1(O, x)
        Z = self.ffn(x)
        x = self.add2(Z, x)
        return x

class Encoder(nn.Module):
    def __init__(self, word_size, hidden_size, num_heads, num_layers):
        super().__init__()
        self.posit = DynamicalPositionalEncoding(word_size)
        self.layers = nn.ModuleList(
            [EncoderBlock(word_size, hidden_size, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, src_attention_mask=None):
        x = self.posit(x)
        for layer in self.layers:
            x = layer(x, src_attention_mask)
        return x
