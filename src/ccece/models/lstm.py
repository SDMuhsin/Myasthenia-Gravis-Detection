"""
CCECE Paper: LSTM Variants for Time Series Classification

Includes vanilla LSTM and Bidirectional LSTM with attention.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .base import BaseTimeSeriesModel, Attention


class VanillaLSTM(BaseTimeSeriesModel):
    """
    Vanilla LSTM for time series classification.

    Architecture:
        Input -> LSTM -> Last hidden state -> FC -> Output

    Simple baseline for comparison with more complex models.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs,  # Accept extra arguments from experiment runner
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            **kwargs: Extra arguments (ignored, for compatibility)
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

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
        # LSTM: (batch_size, seq_len, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state: (batch_size, hidden_dim)
        # h_n shape is (num_layers, batch_size, hidden_dim)
        last_hidden = h_n[-1]

        # Classification
        logits = self.classifier(last_hidden)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'VanillaLSTM',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
        }


class BiLSTMAttention(BaseTimeSeriesModel):
    """
    Bidirectional LSTM with Attention mechanism.

    Architecture:
        Input -> BiLSTM -> Attention -> FC -> Output

    Similar to BiGRU+Attention but uses LSTM cells for comparison.
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
        **kwargs,  # Accept extra arguments from experiment runner
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            hidden_dim: LSTM hidden dimension (per direction)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            fc_dim: Fully connected layer dimension
            **kwargs: Extra arguments (ignored, for compatibility)
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.fc_dim = fc_dim

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention over LSTM outputs (hidden_dim * 2 for bidirectional)
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
        # BiLSTM: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # Attention: (batch_size, hidden_dim * 2)
        context, attention_weights = self.attention(lstm_out)

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
        with torch.no_grad():
            self.eval()
            _ = self.forward(x)
            return self._attention_weights

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'BiLSTMAttention',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'fc_dim': self.fc_dim,
        }


# Default configurations
VANILLA_LSTM_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.3,
}

BILSTM_ATTENTION_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.3,
    'fc_dim': 64,
}
