from torch import nn
from .components import MultiHeadAttention, LayerNormalization, Add, FeedForward, DynamicalPositionalEncoding

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads):
        super().__init__()
        self.attent = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNormalization(embed_dim)
        self.norm2 = LayerNormalization(embed_dim)
        self.add1 = Add(self.norm1)
        self.add2 = Add(self.norm2)
        self.ffn = FeedForward(embed_dim, hidden_size)

    def forward(self, x, src_attention_mask=None):
        O = self.attent(x, x)
        if src_attention_mask is not None:
            O = O.masked_fill(src_attention_mask == 0, float('-inf'))
        x = self.add1(O, x)
        Z = self.ffn(x)
        x = self.add2(Z, x)
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, num_layers):
        super().__init__()
        self.posit = DynamicalPositionalEncoding(embed_dim)
        self.layers = nn.ModuleList(
            [EncoderBlock(embed_dim, hidden_size, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, src_attention_mask=None):
        x = self.posit(x)
        for layer in self.layers:
            x = layer(x, src_attention_mask)
        return x
