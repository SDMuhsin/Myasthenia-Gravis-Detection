"""
CCECE Paper: Bidirectional GRU with Attention

This is the baseline model that achieved 72.36% accuracy in Experiment 13.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .base import BaseTimeSeriesModel, Attention


class BiGRUAttention(BaseTimeSeriesModel):
    """
    Bidirectional GRU with Attention mechanism.

    Architecture:
        Input -> BiGRU -> Attention -> FC -> Output

    This model processes the time series bidirectionally and uses
    attention to focus on the most informative timesteps.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_dim: int = 64,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            hidden_dim: GRU hidden dimension (per direction)
            num_layers: Number of GRU layers
            dropout: Dropout probability
            fc_dim: Fully connected layer dimension
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.fc_dim = fc_dim

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention over GRU outputs (hidden_dim * 2 for bidirectional)
        self.attention = Attention(hidden_dim * 2, seq_len)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

        # Store attention weights for explainability
        self._attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # BiGRU: (batch_size, seq_len, hidden_dim * 2)
        gru_out, _ = self.gru(x)

        # Attention: (batch_size, hidden_dim * 2)
        context, attention_weights = self.attention(gru_out)

        # Store for explainability
        self._attention_weights = attention_weights.detach()

        # Classification
        logits = self.classifier(context)

        return logits

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights for the input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Attention weights of shape (batch_size, seq_len)
        """
        # Run forward pass to compute attention weights
        with torch.no_grad():
            self.eval()
            _ = self.forward(x)
            return self._attention_weights

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'BiGRUAttention',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'fc_dim': self.fc_dim,
        }


# Default configuration that matches Experiment 13 baseline
DEFAULT_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.3,
    'fc_dim': 64,
}
