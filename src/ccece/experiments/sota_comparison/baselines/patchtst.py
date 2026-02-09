"""
PatchTST Wrapper for SOTA Comparison

PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
Segments time series into patches and uses Transformer for classification.

Reference:
    Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)

Implementation Notes:
    - Adaptive patch sizing based on sequence length (target ~12-16 patches)
    - Uses channel-independent processing as per the original paper
    - Instance normalization (RevIN) for better generalization
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .base import PyTorchBaselineModel


def compute_adaptive_patch_params(seq_len: int, min_patch_len: int = 8, max_patch_len: int = 64) -> Tuple[int, int]:
    """
    Compute adaptive patch_len and stride based on sequence length.

    The original PatchTST paper tunes patch_len as a hyperparameter per dataset.
    This function provides a reasonable default that targets ~12-16 patches.

    Args:
        seq_len: Length of input sequence
        min_patch_len: Minimum patch length (default 8)
        max_patch_len: Maximum patch length (default 64)

    Returns:
        Tuple of (patch_len, stride) where stride = patch_len // 2 (50% overlap)

    Reference:
        Nie et al. ICLR 2023 - patch_len is tuned per dataset, typically 8-64
    """
    # Target ~12-16 patches for effective attention
    target_patches = 14

    # Calculate ideal patch_len to achieve target_patches with 50% overlap
    # n_patches = (seq_len - patch_len) // stride + 1
    # With stride = patch_len // 2:
    # n_patches ≈ 2 * seq_len / patch_len - 1
    # Solving: patch_len ≈ 2 * seq_len / (n_patches + 1)

    ideal_patch_len = int(2 * seq_len / (target_patches + 1))

    # Clamp to reasonable bounds
    patch_len = max(min_patch_len, min(ideal_patch_len, max_patch_len))

    # Round to nearest power of 2 for efficiency (optional but common practice)
    # patch_len = 2 ** int(round(math.log2(patch_len)))

    # Stride is 50% overlap as per paper
    stride = max(1, patch_len // 2)

    return patch_len, stride


class PatchEmbedding(nn.Module):
    """
    Patch Embedding for time series.

    Converts multivariate time series into sequence of patches,
    then projects each patch to embedding dimension.
    """

    def __init__(
        self,
        input_dim: int,
        patch_len: int,
        stride: int,
        d_model: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Project patch to d_model
        self.patch_projection = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, n_patches, d_model)
        """
        batch_size, seq_len, input_dim = x.shape

        # Create patches by unfolding
        # First transpose to (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Unfold to get patches: (batch, input_dim, n_patches, patch_len)
        x_patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # Reshape: (batch, n_patches, input_dim * patch_len)
        n_patches = x_patches.shape[2]
        x_patches = x_patches.permute(0, 2, 1, 3)  # (batch, n_patches, input_dim, patch_len)
        x_patches = x_patches.reshape(batch_size, n_patches, -1)

        # Project to d_model
        x_embedded = self.patch_projection(x_patches)  # (batch, n_patches, d_model)

        return x_embedded


class PatchTSTWrapper(PyTorchBaselineModel):
    """
    PatchTST: Patch Time Series Transformer for classification.

    Key innovations:
    1. Patching: Segments time series into subseries patches
    2. Channel-independence: Each channel processed independently
    3. Instance normalization: Normalizes each input instance
    4. **Adaptive patch sizing**: Automatically adjusts patch_len and stride
       based on sequence length to maintain ~12-16 patches (NEW)

    Reference:
        Nie et al. "A Time Series is Worth 64 Words" (ICLR 2023)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        seq_len: int = 290,
        patch_len: Optional[int] = None,  # None = adaptive
        stride: Optional[int] = None,  # None = adaptive
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, seq_len, device)

        # Compute adaptive patch parameters if not provided
        if patch_len is None or stride is None:
            computed_patch_len, computed_stride = compute_adaptive_patch_params(seq_len)
            patch_len = patch_len if patch_len is not None else computed_patch_len
            stride = stride if stride is not None else computed_stride

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout

        # Calculate number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding
        self.patch_embed = PatchEmbedding(input_dim, patch_len, stride, d_model)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Instance normalization (RevIN-style)
        self.instance_norm = nn.InstanceNorm1d(input_dim, affine=True)

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
        batch_size = x.shape[0]

        # Instance normalization
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.instance_norm(x)
        x = x.transpose(1, 2)  # (batch, seq_len, input_dim)

        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches + 1, d_model)

        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer encoding
        x = self.transformer(x)  # (batch, n_patches + 1, d_model)

        # Use CLS token for classification
        cls_output = x[:, 0, :]  # (batch, d_model)

        # Classification
        logits = self.classifier(cls_output)

        return logits

    def get_model_name(self) -> str:
        return "PatchTST"
