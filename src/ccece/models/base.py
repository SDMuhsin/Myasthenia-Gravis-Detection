"""
CCECE Paper: Base Model Class

All time series models should inherit from this base class.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseTimeSeriesModel(nn.Module, ABC):
    """
    Abstract base class for time series classification models.

    All models must implement:
        - forward(x): Forward pass
        - get_config(): Return model configuration dict

    Models may optionally implement:
        - get_attention_weights(x): For explainability
    """

    def __init__(self, input_dim: int, num_classes: int, seq_len: int):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes
            seq_len: Expected sequence length (for models that need it)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for reproducibility.

        Returns:
            Dict containing all hyperparameters needed to recreate the model
        """
        pass

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights for explainability (optional).

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Attention weights of shape (batch_size, seq_len) or None
        """
        return None

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> str:
        """Get a summary string of the model."""
        config = self.get_config()
        param_count = self.count_parameters()
        lines = [
            f"Model: {self.__class__.__name__}",
            f"Parameters: {param_count:,}",
            "Config:",
        ]
        for k, v in config.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class Attention(nn.Module):
    """
    Attention mechanism for sequence models.

    Computes attention weights over the sequence dimension and returns
    a weighted sum of the hidden states.
    """

    def __init__(self, hidden_dim: int, seq_len: int):
        """
        Args:
            hidden_dim: Dimension of hidden states
            seq_len: Sequence length
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Attention weight vector
        self.attention_weight = nn.Parameter(torch.zeros(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attention_weight)

        # Bias for attention scores
        self.attention_bias = nn.Parameter(torch.zeros(seq_len))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.

        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)

        Returns:
            context: Weighted sum, shape (batch_size, hidden_dim)
            attention_weights: Shape (batch_size, seq_len)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute attention scores: (batch_size, seq_len)
        # Reshape to (batch_size * seq_len, hidden_dim) for matmul
        scores = torch.mm(
            hidden_states.contiguous().view(-1, hidden_dim),
            self.attention_weight
        ).view(batch_size, seq_len)

        # Add bias and apply tanh
        scores = torch.tanh(scores + self.attention_bias)

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=1)

        # Weighted sum of hidden states
        context = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)

        return context, attention_weights
