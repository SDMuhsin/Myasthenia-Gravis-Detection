"""
InceptionTime Wrapper for SOTA Comparison

Wraps the existing InceptionTime implementation to fit the baseline interface.

Reference:
    Fawaz et al. "InceptionTime: Finding AlexNet for Time Series Classification" (DMKD 2020)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import numpy as np

from .base import PyTorchBaselineModel


class InceptionModule(nn.Module):
    """Inception module with multiple parallel convolution paths."""

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 32,
        kernel_sizes: list = [10, 20, 40],
        bottleneck_channels: int = 32,
    ):
        super().__init__()

        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )

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

        total_filters = num_filters * len(kernel_sizes)
        self.bn = nn.BatchNorm1d(total_filters)
        self.out_channels = total_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bn = self.bottleneck(x)
        outputs = [conv(x_bn) for conv in self.convs]
        min_len = min(o.size(2) for o in outputs)
        outputs = [o[:, :, :min_len] for o in outputs]
        out = torch.cat(outputs, dim=1)
        out = self.bn(out)
        out = torch.relu(out)
        return out


class InceptionTimeWrapper(PyTorchBaselineModel):
    """
    InceptionTime for time series classification.

    Reference:
        Fawaz et al. "InceptionTime: Finding AlexNet for Time Series Classification" (2020)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        seq_len: int = 290,
        num_filters: int = 32,
        depth: int = 6,
        kernel_sizes: list = None,
        bottleneck_channels: int = 32,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, seq_len, device)

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
            module = InceptionModule(
                in_channels=current_channels,
                num_filters=num_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
            )
            self.inception_modules.append(module)

            self.residual_convs.append(
                nn.Conv1d(current_channels, module.out_channels, kernel_size=1, bias=False)
            )

            current_channels = module.out_channels

        self.final_channels = current_channels
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.final_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        for inception, residual_conv in zip(self.inception_modules, self.residual_convs):
            inception_out = inception(x)
            residual = residual_conv(x)

            min_len = min(inception_out.size(2), residual.size(2))
            inception_out = inception_out[:, :, :min_len]
            residual = residual[:, :, :min_len]

            x = inception_out + residual

        x = x.mean(dim=2)  # Global average pooling
        x = self.dropout(x)
        logits = self.fc(x)

        return logits

    def get_model_name(self) -> str:
        return "InceptionTime"
