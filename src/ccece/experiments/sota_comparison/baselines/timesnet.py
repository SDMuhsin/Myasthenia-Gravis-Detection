"""
TimesNet Wrapper for SOTA Comparison

TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.
Converts 1D time series to 2D tensors based on period analysis.

Reference:
    Wu et al. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from typing import Optional

from .base import PyTorchBaselineModel


class Inception_Block_V1(nn.Module):
    """Inception block for 2D processing."""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = [1, 3, 5, 7, 9, 11][:num_kernels]

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1), padding=(k // 2, 0))
            for k in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [conv(x) for conv in self.convs]
        return sum(outputs) / len(outputs)


class TimesBlock(nn.Module):
    """
    TimesBlock: Core building block of TimesNet.

    Performs:
    1. FFT to find dominant periods
    2. Reshape 1D to 2D based on periods
    3. 2D convolution (inception block)
    4. Reshape back to 1D
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int,
        top_k: int = 5,
        num_kernels: int = 6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.top_k = top_k

        # Inception block for 2D processing
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # FFT to find periods
        # Average over channels for period finding
        x_freq = torch.fft.rfft(x.mean(dim=-1), dim=1)  # (batch, seq_len//2+1)
        frequency = abs(x_freq)

        # Find top-k periods (exclude DC component)
        frequency[:, 0] = 0  # Zero out DC
        _, top_indices = torch.topk(frequency, min(self.top_k, frequency.shape[1]), dim=1)

        # Convert frequency indices to periods
        # period = seq_len / frequency_index
        periods = []
        for i in range(self.top_k):
            idx = top_indices[:, i]
            period = (seq_len / (idx.float() + 1e-8)).clamp(min=2, max=seq_len).int()
            periods.append(period)

        # Process each period separately and aggregate
        outputs = []
        for period in periods:
            # Use the most common period in the batch for simplicity
            p = int(period.float().mean().item())
            p = max(2, min(p, seq_len))

            # Compute number of full periods
            n_periods = seq_len // p

            # Reshape to 2D: (batch, d_model, n_periods, p)
            if n_periods > 0:
                x_2d = x[:, :n_periods * p, :].reshape(batch, n_periods, p, d_model)
                x_2d = x_2d.permute(0, 3, 1, 2)  # (batch, d_model, n_periods, p)

                # 2D convolution
                out_2d = self.conv(x_2d)  # (batch, d_model, n_periods, p)

                # Reshape back to 1D
                out_2d = out_2d.permute(0, 2, 3, 1)  # (batch, n_periods, p, d_model)
                out = out_2d.reshape(batch, n_periods * p, d_model)

                # Pad if necessary
                if out.shape[1] < seq_len:
                    out = F.pad(out, (0, 0, 0, seq_len - out.shape[1]))
                else:
                    out = out[:, :seq_len, :]

                outputs.append(out)

        if outputs:
            # Aggregate outputs
            result = sum(outputs) / len(outputs)
        else:
            result = x

        return result + x  # Residual connection


class TimesNetWrapper(PyTorchBaselineModel):
    """
    TimesNet for time series classification.

    Reference:
        Wu et al. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        seq_len: int = 290,
        d_model: int = 64,
        d_ff: int = 64,
        n_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, seq_len, device)

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.top_k = top_k
        self.dropout_rate = dropout

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, d_ff, top_k, num_kernels)
            for _ in range(n_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Logits (batch, num_classes)
        """
        # Embed
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.dropout(x)

        # TimesNet blocks
        for block, ln in zip(self.blocks, self.layer_norms):
            x = block(x)
            x = ln(x)
            x = self.dropout(x)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)

        return logits

    def get_model_name(self) -> str:
        return "TimesNet"
