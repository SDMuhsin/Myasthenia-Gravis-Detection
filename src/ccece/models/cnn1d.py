"""
CCECE Paper: 1D Convolutional Neural Network for Time Series Classification

Simple and fast baseline using 1D convolutions for feature extraction.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .base import BaseTimeSeriesModel


class CNN1D(BaseTimeSeriesModel):
    """
    1D Convolutional Neural Network for time series classification.

    Architecture:
        Input -> Conv blocks -> Global Pooling -> FC -> Output

    Features:
        - Multiple conv layers with increasing channels
        - Batch normalization for training stability
        - Global average pooling for sequence-length invariance
        - Very fast compared to RNNs
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
        **kwargs,  # Accept extra arguments from experiment runner
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            hidden_dim: Number of channels in conv layers
            num_layers: Number of conv blocks
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            **kwargs: Extra arguments (ignored, for compatibility)
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Build conv blocks
        layers = []
        in_channels = input_dim

        for i in range(num_layers):
            out_channels = hidden_dim * (2 ** min(i, 2))  # Cap growth at 4x

            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.final_channels = in_channels

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.final_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Transpose for conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Conv layers: (batch_size, final_channels, reduced_seq_len)
        x = self.conv_layers(x)

        # Global average pooling: (batch_size, final_channels)
        x = x.mean(dim=2)

        # Classification
        logits = self.classifier(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'CNN1D',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
        }


# Default configuration
DEFAULT_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 3,
    'kernel_size': 7,
    'dropout': 0.2,
}
