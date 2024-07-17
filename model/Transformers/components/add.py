from torch import nn
class Add(nn.Module):
    def __init__(self, activation=None):
        super().__init__()
        self.activation = activation if activation else lambda x: x

    def forward(self, x, y):
        Z = x + y
        Z = self.activation(Z)
        return Z
