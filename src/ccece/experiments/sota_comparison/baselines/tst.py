"""
Time Series Transformer (TST) for SOTA Comparison

A Transformer-based framework for multivariate time series representation learning.

Reference:
    Zerveas et al. "A Transformer-based Framework for Multivariate Time Series
    Representation Learning" (KDD 2021)

Implementation Notes (faithful to paper):
    - d_model: 128-512 (paper uses larger for longer sequences)
    - n_heads: 8 (paper default)
    - n_layers: 3-6 (paper uses deeper for complex tasks)
    - d_ff: 256-1024 (paper uses 2-4x d_model)
    - Added optional sequence downsampling for long sequences (>512)
      to handle O(n²) attention cost
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .base import PyTorchBaselineModel


def compute_downsample_factor(seq_len: int, max_seq_len: int = 512) -> int:
    """
    Compute downsampling factor for long sequences.

    For sequences longer than max_seq_len, we downsample to reduce
    the O(n²) cost of self-attention.

    Args:
        seq_len: Original sequence length
        max_seq_len: Maximum sequence length before downsampling

    Returns:
        Downsampling factor (1 if no downsampling needed)
    """
    if seq_len <= max_seq_len:
        return 1
    # Target max_seq_len tokens
    return max(1, (seq_len + max_seq_len - 1) // max_seq_len)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(PyTorchBaselineModel):
    """
    Time Series Transformer (TST) for multivariate time series classification.

    Architecture:
        Input (batch, seq_len, input_dim)
            |
        [Optional] Sequence downsampling (for seq_len > max_seq_len)
            |
        Linear projection to d_model
            |
        Positional encoding
            |
        Transformer encoder (N layers)
            |
        Global average pooling
            |
        FC classifier

    Reference:
        Zerveas et al. "A Transformer-based Framework for Multivariate Time Series
        Representation Learning" (KDD 2021)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        seq_len: int = 290,
        d_model: int = 256,       # Paper: 128-512 (increased from 128)
        n_heads: int = 8,         # Paper: 8 (increased from 4)
        n_layers: int = 4,        # Paper: 3-6 (increased from 3)
        d_ff: int = 512,          # Paper: 2-4x d_model (increased from 256)
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 512,   # Downsample sequences longer than this
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, seq_len, device)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len

        # Compute downsampling factor for long sequences
        self.downsample_factor = compute_downsample_factor(seq_len, max_seq_len)
        self.effective_seq_len = (seq_len + self.downsample_factor - 1) // self.downsample_factor

        # Downsampling layer (strided convolution for long sequences)
        if self.downsample_factor > 1:
            self.downsample = nn.Conv1d(
                input_dim, input_dim,
                kernel_size=self.downsample_factor * 2 + 1,
                stride=self.downsample_factor,
                padding=self.downsample_factor,
                groups=input_dim,  # Depthwise convolution
            )
        else:
            self.downsample = None

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.effective_seq_len + 100, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

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
        # Apply downsampling for long sequences
        if self.downsample is not None:
            # (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
            x = x.transpose(1, 2)
            x = self.downsample(x)
            # (batch, input_dim, reduced_seq_len) -> (batch, reduced_seq_len, input_dim)
            x = x.transpose(1, 2)

        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)

        return logits

    def get_model_name(self) -> str:
        return "TST"
