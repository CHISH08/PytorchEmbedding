from torch import nn
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.ffn_layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim)
        )

    def forward(self, x):
        Z = self.ffn_layers(x)
        return Z
