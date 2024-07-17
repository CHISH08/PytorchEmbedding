from torch import nn
class FeedForward(nn.Module):
    def __init__(self, word_size, hidden_size):
        super().__init__()
        self.ffn_layers = nn.Sequential(
            nn.Linear(word_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, word_size)
        )

    def forward(self, x):
        Z = self.ffn_layers(x)
        return Z
