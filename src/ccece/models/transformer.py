"""
CCECE Paper: Transformer for Time Series Classification

Uses self-attention to capture dependencies at any distance in the sequence.
Includes positional encoding since attention is permutation-invariant.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any

from .base import BaseTimeSeriesModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size, seq_len, d_model)
        Returns:
            Shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(BaseTimeSeriesModel):
    """
    Transformer-based time series classifier.

    Architecture:
        Input -> Linear projection -> Positional Encoding -> Transformer Encoder -> Global Pooling -> FC -> Output

    Uses a smaller variant suitable for the dataset size and computational constraints.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_dim: int = 128,
        **kwargs,  # Accept extra arguments from experiment runner
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            hidden_dim: Embedding dimension (d_model)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            ff_dim: Feed-forward dimension in transformer
            **kwargs: Extra arguments (ignored, for compatibility)
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ff_dim = ff_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len + 100, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
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
        # Project input: (batch_size, seq_len, hidden_dim)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder: (batch_size, seq_len, hidden_dim)
        x = self.transformer_encoder(x)

        # Global average pooling: (batch_size, hidden_dim)
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'TransformerClassifier',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'ff_dim': self.ff_dim,
        }


# Default configuration
DEFAULT_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'num_heads': 4,
    'dropout': 0.1,
    'ff_dim': 128,
}
