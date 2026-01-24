"""
Simple 1D-CNN Baseline

A standard 3-layer 1D convolutional neural network for time series classification.
This serves as a simple baseline to establish that the task is non-trivial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import PyTorchBaselineModel


class SimpleCNN(PyTorchBaselineModel):
    """
    Standard 3-layer 1D CNN classifier.

    Architecture:
        Input (batch, seq_len, input_dim)
            |
        Conv1d(64, k=7) -> BN -> ReLU -> MaxPool
            |
        Conv1d(128, k=5) -> BN -> ReLU -> MaxPool
            |
        Conv1d(256, k=3) -> BN -> ReLU -> AdaptiveAvgPool
            |
        FC(256) -> ReLU -> Dropout -> FC(num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        seq_len: int = 290,
        hidden_channels: tuple = (64, 128, 256),
        kernel_sizes: tuple = (7, 5, 3),
        dropout: float = 0.3,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, seq_len, device)

        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim

        for i, (out_channels, kernel_size) in enumerate(zip(hidden_channels, kernel_sizes)):
            self.conv_layers.append(
                nn.Sequential(
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
                )
            )
            in_channels = out_channels

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Logits (batch, num_classes)
        """
        # Transpose for conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Global pooling
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)

        # Classifier
        logits = self.classifier(x)

        return logits

    def get_model_name(self) -> str:
        return "1D-CNN"
