"""CNN model for MNIST classification.

A simple but effective CNN architecture with two conv blocks followed by
a fully-connected head. The penultimate FC layer serves as the embedding
layer for downstream visualization.
"""

import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    """Basic CNN for MNIST digit classification.

    Architecture:
        Conv(1→32, 3x3) → ReLU → Conv(32→64, 3x3) → ReLU → MaxPool → Dropout
        → Flatten → FC(9216→128) → ReLU → Dropout → FC(128→num_classes)

    The 128-dim FC layer is the embedding layer used for UMAP projections.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc1_relu = nn.ReLU()
        self.fc1_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 128-dim penultimate layer activations."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        return x
