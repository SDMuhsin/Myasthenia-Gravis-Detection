"""
ConvTran Wrapper for SOTA Comparison

ConvTran: Improving Position Encoding of Transformers for Multivariate Time Series Classification.
Uses convolutional positional encoding instead of traditional sinusoidal encoding.

Reference:
    Foumani et al. "Improving Position Encoding of Transformers for Multivariate
    Time Series Classification" (DMKD 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .base import PyTorchBaselineModel


class ConvPositionalEncoding(nn.Module):
    """
    Convolutional Positional Encoding.

    Uses 1D convolution to learn position-aware representations,
    which is more effective than sinusoidal encoding for time series.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,  # Depthwise convolution
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # Transpose for conv1d
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)

        # Convolutional positional encoding
        pos = self.conv(x_t)  # (batch, d_model, seq_len)
        pos = pos.transpose(1, 2)  # (batch, seq_len, d_model)

        # Add to input and normalize
        out = self.ln(x + self.dropout(pos))

        return out


class ConvTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with convolutional positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        conv_kernel_size: int = 3,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.conv_pos = ConvPositionalEncoding(d_model, conv_kernel_size, dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # Convolutional positional encoding
        x = self.conv_pos(x)

        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class ConvTranWrapper(PyTorchBaselineModel):
    """
    ConvTran: Convolutional Transformer for time series classification.

    Key innovation: Uses convolutional positional encoding instead of
    sinusoidal encoding, which better captures local temporal patterns.

    Reference:
        Foumani et al. "Improving Position Encoding of Transformers for
        Multivariate Time Series Classification" (DMKD 2023)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        seq_len: int = 290,
        d_model: int = 64,  # Reduced for memory efficiency
        n_heads: int = 4,
        n_layers: int = 2,  # Reduced layers
        d_ff: int = 128,  # Reduced for memory efficiency
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 512,  # Maximum sequence length for attention
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

        # Calculate stride needed to reduce sequence length
        self.stride = max(1, (seq_len + max_seq_len - 1) // max_seq_len)
        self.reduced_seq_len = (seq_len + self.stride - 1) // self.stride

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Sequence reduction via strided convolution (for long sequences)
        # This reduces seq_len before attention to avoid O(n^2) memory issues
        self.seq_reduction = nn.Conv1d(
            d_model, d_model,
            kernel_size=self.stride * 2 + 1,
            stride=self.stride,
            padding=self.stride,
        )

        # Initial convolutional embedding
        self.conv_embed = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # ConvTransformer layers
        self.layers = nn.ModuleList([
            ConvTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size,
            )
            for _ in range(n_layers)
        ])

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
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Sequence reduction for long sequences
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        if self.stride > 1:
            x_t = self.seq_reduction(x_t)  # (batch, d_model, reduced_seq_len)

        # Initial conv embedding
        x_t = self.conv_embed(x_t)
        x = x_t.transpose(1, 2)  # (batch, reduced_seq_len, d_model)

        # ConvTransformer layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)

        return logits

    def get_model_name(self) -> str:
        return "ConvTran"
