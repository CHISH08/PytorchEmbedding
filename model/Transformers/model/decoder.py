from torch import nn
from .components import MultiHeadAttention, LayerNormalization, Add, FeedForward, DynamicalPositionalEncoding, MaskedMultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads):
        super().__init__()
        self.mask = MaskedMultiHeadAttention(embed_dim, num_heads)
        self.attent = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_size)
        self.norm1 = LayerNormalization(embed_dim)
        self.norm2 = LayerNormalization(embed_dim)
        self.norm3 = LayerNormalization(embed_dim)
        self.add1 = Add(self.norm1)
        self.add2 = Add(self.norm2)
        self.add3 = Add(self.norm3)

    def forward(self, x, encoder_output, memory_key_padding_mask=None):
        O = self.mask(x)
        x = self.add1(O, x)
        O = self.attent(x, encoder_output)
        if memory_key_padding_mask is not None:
            O = O.masked_fill(memory_key_padding_mask == 0, float('-inf'))
        x = self.add2(O, x)
        Z = self.ffn(x)
        x = self.add3(Z, x)
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, num_layers):
        super().__init__()
        self.posit = DynamicalPositionalEncoding(embed_dim)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_dim, hidden_size, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, encoder_output, memory_key_padding_mask=None):
        x = self.posit(x)
        for layer in self.layers:
            x = layer(x, encoder_output, memory_key_padding_mask)
        return x
