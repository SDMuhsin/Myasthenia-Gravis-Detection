"""
Simple LSTM Baseline

A standard 2-layer bidirectional LSTM for time series classification.
This serves as a simple baseline to establish that the task is non-trivial.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import PyTorchBaselineModel


class SimpleLSTM(PyTorchBaselineModel):
    """
    Standard 2-layer BiLSTM classifier.

    Architecture:
        Input (batch, seq_len, input_dim)
            |
        BiLSTM(hidden_dim, 2 layers)
            |
        Last hidden state (batch, hidden_dim * 2)
            |
        FC(hidden_dim) -> ReLU -> Dropout -> FC(num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        seq_len: int = 290,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, seq_len, device)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classifier head
        lstm_output_dim = hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Logits (batch, num_classes)
        """
        batch_size = x.size(0)

        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from both directions
        if self.bidirectional:
            # h_n[-2] is forward final, h_n[-1] is backward final
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_dim * 2)
        else:
            h_final = h_n[-1]  # (batch, hidden_dim)

        # Classifier
        logits = self.classifier(h_final)

        return logits

    def get_model_name(self) -> str:
        return "LSTM"
