"""
CCECE Paper: InceptionTime for Time Series Classification

InceptionTime is an ensemble of Inception-based CNNs that achieves state-of-the-art
results on the UCR time series classification archive.

Reference:
    Fawaz et al. "InceptionTime: Finding AlexNet for Time Series Classification" (2020)
    https://arxiv.org/abs/1909.04939
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseTimeSeriesModel


class InceptionModule(nn.Module):
    """
    Inception module with multiple parallel convolution paths.
    Simplified version without max pooling branch for stability.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 32,
        kernel_sizes: List[int] = [10, 20, 40],
        bottleneck_channels: int = 32,
    ):
        super().__init__()

        # Bottleneck to reduce dimensionality
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )

        # Parallel convolutions with different kernel sizes
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(
                nn.Conv1d(
                    bottleneck_channels,
                    num_filters,
                    kernel_size=ks,
                    padding=ks // 2,
                    bias=False,
                )
            )

        # Batch normalization
        total_filters = num_filters * len(kernel_sizes)
        self.bn = nn.BatchNorm1d(total_filters)

        self.out_channels = total_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, seq_len)
        Returns:
            (batch_size, out_channels, seq_len)
        """
        # Bottleneck
        x_bn = self.bottleneck(x)

        # Parallel convolutions
        outputs = [conv(x_bn) for conv in self.convs]

        # Ensure same length (handle odd kernel sizes)
        min_len = min(o.size(2) for o in outputs)
        outputs = [o[:, :, :min_len] for o in outputs]

        # Concatenate
        out = torch.cat(outputs, dim=1)
        out = self.bn(out)
        out = F.relu(out)

        return out


class InceptionTime(BaseTimeSeriesModel):
    """
    InceptionTime: Inception-based CNN for time series classification.

    Architecture:
        Input -> Inception Modules (with residuals) -> Global Pool -> FC -> Output

    Key Features:
        - Multi-scale temporal pattern extraction via inception modules
        - Residual connections for gradient flow
        - Bottleneck layers for efficiency
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        num_filters: int = 32,
        depth: int = 6,
        kernel_sizes: List[int] = None,
        bottleneck_channels: int = 32,
        dropout: float = 0.3,
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes
            seq_len: Expected sequence length
            num_filters: Number of filters per inception path
            depth: Number of inception modules
            kernel_sizes: List of kernel sizes for inception modules
            bottleneck_channels: Channels in bottleneck layer
            dropout: Dropout probability
        """
        super().__init__(input_dim, num_classes, seq_len)

        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]

        self.num_filters = num_filters
        self.depth = depth
        self.kernel_sizes = kernel_sizes
        self.bottleneck_channels = bottleneck_channels
        self.dropout_rate = dropout

        # Build inception modules
        self.inception_modules = nn.ModuleList()
        self.residual_convs = nn.ModuleList()

        current_channels = input_dim
        for i in range(depth):
            # Inception module
            module = InceptionModule(
                in_channels=current_channels,
                num_filters=num_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
            )
            self.inception_modules.append(module)

            # Residual connection (1x1 conv to match channels)
            self.residual_convs.append(
                nn.Conv1d(current_channels, module.out_channels, kernel_size=1, bias=False)
            )

            current_channels = module.out_channels

        self.final_channels = current_channels

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

        # Inception modules with residual connections
        for i, (inception, residual_conv) in enumerate(zip(self.inception_modules, self.residual_convs)):
            # Compute inception output
            inception_out = inception(x)

            # Compute residual (project to match channels)
            residual = residual_conv(x)

            # Ensure same length
            min_len = min(inception_out.size(2), residual.size(2))
            inception_out = inception_out[:, :, :min_len]
            residual = residual[:, :, :min_len]

            # Add residual
            x = inception_out + residual

        # Global average pooling: (batch_size, channels)
        x = x.mean(dim=2)

        # Classification
        x = self.dropout(x)
        logits = self.fc(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'InceptionTime',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'num_filters': self.num_filters,
            'depth': self.depth,
            'kernel_sizes': self.kernel_sizes,
            'bottleneck_channels': self.bottleneck_channels,
            'dropout': self.dropout_rate,
        }


# Default configuration
DEFAULT_CONFIG = {
    'num_filters': 32,
    'depth': 6,
    'kernel_sizes': [10, 20, 40],
    'bottleneck_channels': 32,
    'dropout': 0.2,
}
