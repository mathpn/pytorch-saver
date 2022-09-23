"""
Dummy PyTorch model used for testing.

But feel free to use it if you want to.
"""

from torch import nn


class Model(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, n_layers: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        hidden_layers = []
        for _ in range(n_layers - 1):
            hidden_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
        hidden_layers.extend(
            [
                nn.Linear(hidden_dim, n_classes),
            ]
        )
        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, x):
        x = self.fc1(x)
        x = self.hidden_layers(x)
        return x
