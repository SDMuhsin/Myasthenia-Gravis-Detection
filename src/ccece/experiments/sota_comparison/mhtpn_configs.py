"""
MHTPN Configuration Module for Multi-Dataset SOTA Comparison

Provides per-dataset model configurations and standardized training parameters.
These configurations are optimized based on validation experiments and must
remain consistent for fair comparison.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .datasets import compute_optimal_n_segments, get_dataset_info


@dataclass
class MHTPNModelConfig:
    """Model architecture configuration for MHTPN."""
    latent_dim: int = 64
    n_heads: int = 5
    head_dim: int = 32
    n_segments: int = 8
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2


@dataclass
class MHTPNTrainingConfig:
    """Training configuration for MHTPN.

    These parameters are standardized across all datasets for fair comparison.
    Based on extensive validation that showed:
    - epochs=150 allows better convergence (especially for EEG/ECG data)
    - patience=30 prevents premature early stopping
    - cosine annealing improves final accuracy
    """
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 30
    use_cosine_annealing: bool = True
    grad_clip_norm: float = 1.0
    # Loss weights for prototype learning
    cluster_loss_weight: float = 0.3
    separation_loss_weight: float = 0.1
    diversity_loss_weight: float = 0.05


# Default training config (MUST be same for all datasets)
TRAINING_CONFIG = MHTPNTrainingConfig()


# Per-dataset model configurations
# Note: Only n_segments varies based on sequence length
MHTPN_MODEL_CONFIGS: Dict[str, MHTPNModelConfig] = {
    'MG': MHTPNModelConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=32,
        n_segments=8,  # seq_len=2903, segment_size=363
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
    ),
    'Heartbeat': MHTPNModelConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=32,
        n_segments=4,  # seq_len=405, segment_size=101
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
    ),
    'BasicMotions': MHTPNModelConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=32,
        n_segments=2,  # seq_len=100, segment_size=50
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
    ),
    'Epilepsy': MHTPNModelConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=32,
        n_segments=2,  # seq_len=206, segment_size=103
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
    ),
    'SpokenArabicDigits': MHTPNModelConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=32,
        n_segments=2,  # seq_len=65, segment_size=33 (audio MFCC features)
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
    ),
    'LSST': MHTPNModelConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=32,
        n_segments=2,  # seq_len=36, segment_size=18 (astronomy light curves)
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
    ),
}


def get_mhtpn_model_config(
    dataset_name: str,
    seq_len: Optional[int] = None,
) -> MHTPNModelConfig:
    """
    Get MHTPN model configuration for a dataset.

    Args:
        dataset_name: Name of dataset
        seq_len: Optional sequence length (used to auto-compute n_segments
                 if dataset not in predefined configs)

    Returns:
        MHTPNModelConfig for the dataset
    """
    if dataset_name in MHTPN_MODEL_CONFIGS:
        return MHTPN_MODEL_CONFIGS[dataset_name]

    # Fallback: create config with auto-computed n_segments
    if seq_len is None:
        info = get_dataset_info(dataset_name)
        seq_len = info['seq_len']

    n_segments = compute_optimal_n_segments(seq_len)

    return MHTPNModelConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=32,
        n_segments=n_segments,
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
    )


def get_mhtpn_training_config() -> MHTPNTrainingConfig:
    """
    Get standardized training configuration.

    The training config is the same for all datasets to ensure fair comparison.

    Returns:
        MHTPNTrainingConfig with standardized parameters
    """
    return MHTPNTrainingConfig()


def config_to_dict(config: Any) -> Dict[str, Any]:
    """Convert a dataclass config to dictionary."""
    from dataclasses import asdict
    return asdict(config)


def get_full_mhtpn_config(
    dataset_name: str,
    seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get full MHTPN configuration (model + training) as a dictionary.

    Args:
        dataset_name: Name of dataset
        seq_len: Optional sequence length for auto-config

    Returns:
        Dict with 'model' and 'training' sub-dicts
    """
    model_config = get_mhtpn_model_config(dataset_name, seq_len)
    training_config = get_mhtpn_training_config()

    return {
        'model': config_to_dict(model_config),
        'training': config_to_dict(training_config),
    }
