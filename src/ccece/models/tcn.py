"""
CCECE Paper: Temporal Convolutional Network (TCN)

TCN uses dilated causal convolutions to capture long-range dependencies
in time series data efficiently.

Reference: Bai et al., "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling" (2018)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from .base import BaseTimeSeriesModel


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with padding to preserve sequence length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        # Padding to ensure output length equals input length (causal)
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size, channels, seq_len)
        Returns:
            Shape (batch_size, out_channels, seq_len)
        """
        out = self.conv(x)
        # Remove extra padding from the right (causal - no future information)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    A single temporal block with two causal convolutions and a residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection (1x1 conv if channels differ)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size, in_channels, seq_len)
        Returns:
            Shape (batch_size, out_channels, seq_len)
        """
        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TCN(BaseTimeSeriesModel):
    """
    Temporal Convolutional Network for time series classification.

    Architecture:
        Input -> Temporal Blocks (with exponentially increasing dilation) -> Global Pooling -> FC -> Output

    Key features:
        - Dilated causal convolutions for long-range dependencies
        - Residual connections for gradient flow
        - Efficient parallel computation (unlike RNNs)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.2,
        **kwargs,  # Accept extra arguments from experiment runner
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            hidden_dim: Number of channels in temporal blocks
            num_layers: Number of temporal blocks
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            **kwargs: Extra arguments (ignored, for compatibility)
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Build temporal blocks with exponentially increasing dilation
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            out_channels = hidden_dim
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.temporal_blocks = nn.Sequential(*layers)

        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

        # Temporal blocks: (batch_size, hidden_dim, seq_len)
        x = self.temporal_blocks(x)

        # Global average pooling: (batch_size, hidden_dim)
        x = x.mean(dim=2)

        # Classification
        logits = self.classifier(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'TCN',
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
    'num_layers': 4,
    'kernel_size': 7,
    'dropout': 0.2,
}
