"""
CCECE Paper: ResNet1D for Time Series Classification

Deep Residual Networks adapted for 1D time series classification.
Skip connections enable training of deeper networks without vanishing gradients.

Reference:
    Wang et al. "Time Series Classification from Scratch with Deep Neural Networks" (2017)
    https://arxiv.org/abs/1611.06455
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseTimeSeriesModel


class ResidualBlock1D(nn.Module):
    """
    Basic residual block for 1D convolutions.

    Structure: Conv -> BN -> ReLU -> Conv -> BN -> + input -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        stride: int = 1,
        downsample: bool = False,
    ):
        super().__init__()

        self.downsample = downsample

        # First conv
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second conv
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection (with optional downsample)
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Ensure same length for residual addition
        min_len = min(out.size(2), identity.size(2))
        out = out[:, :, :min_len]
        identity = identity[:, :, :min_len]

        out = out + identity
        out = F.relu(out)

        return out


class ResNet1D(BaseTimeSeriesModel):
    """
    ResNet adapted for 1D time series classification.

    Architecture:
        Input -> Initial Conv -> ResBlocks -> Global Pool -> FC -> Output

    Key Features:
        - Skip connections for deep network training
        - Batch normalization for stable training
        - Progressive channel increase
        - Global average pooling for sequence-length invariance
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        kernel_size: int = 8,
        dropout: float = 0.2,
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes
            seq_len: Expected sequence length
            hidden_dim: Base number of channels
            num_blocks: Number of residual block groups
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # Initial convolution
        self.conv1 = nn.Conv1d(
            input_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Residual blocks with progressive channel increase
        self.blocks = nn.ModuleList()
        in_channels = hidden_dim

        for i in range(num_blocks):
            out_channels = hidden_dim * (2 ** min(i, 2))  # 64 -> 128 -> 256
            downsample = (i > 0)  # Downsample after first block

            # Two residual blocks per group
            self.blocks.append(
                ResidualBlock1D(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    stride=2 if downsample else 1,
                    downsample=downsample,
                )
            )
            self.blocks.append(
                ResidualBlock1D(
                    out_channels, out_channels,
                    kernel_size=kernel_size,
                )
            )
            in_channels = out_channels

        self.final_channels = in_channels

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.final_channels, num_classes)

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

        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=2)

        # Classification
        x = self.dropout(x)
        logits = self.fc(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'ResNet1D',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_blocks': self.num_blocks,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
        }


# Default configuration
DEFAULT_CONFIG = {
    'hidden_dim': 64,
    'num_blocks': 3,
    'kernel_size': 8,
    'dropout': 0.2,
}
