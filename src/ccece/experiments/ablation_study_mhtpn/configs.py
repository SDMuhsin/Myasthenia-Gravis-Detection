"""
Ablation Configurations for MultiHeadTrajectoryProtoNet

14 variants across 4 ablation studies:
- Ablation 1: n_segments {1, 4, 8, 16}
- Ablation 2: Trajectory vs Static prototypes
- Ablation 3: Loss component combinations
- Ablation 4: Segment weighting strategies

All variants use the same base configuration except for the ablated component.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class SegmentWeightingStrategy(Enum):
    """Segment weighting strategies for ablation 4."""
    UNIFORM = "uniform"  # All segments weighted equally
    PADDING_AWARE = "padding_aware"  # Weight by fraction of real data (default)
    LEARNED_ATTENTION = "learned_attention"  # Trainable attention weights


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""

    # Identifiers
    ablation_id: str  # e.g., "A1.1"
    ablation_name: str  # e.g., "n_segments"
    variant_name: str  # e.g., "n_segments=1"
    description: str

    # Model hyperparameters
    latent_dim: int = 64
    n_heads: int = 5
    head_dim: int = 32
    n_segments: int = 8
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    grad_clip_norm: float = 1.0

    # Loss weights
    cluster_loss_weight: float = 0.3
    separation_loss_weight: float = 0.1
    diversity_loss_weight: float = 0.05

    # Ablation-specific options
    use_trajectory_prototypes: bool = True  # For ablation 2
    segment_weighting: SegmentWeightingStrategy = SegmentWeightingStrategy.PADDING_AWARE

    # Cross-validation
    n_folds: int = 5

    # Flags
    is_default: bool = False  # True for the default configuration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ablation_id': self.ablation_id,
            'ablation_name': self.ablation_name,
            'variant_name': self.variant_name,
            'description': self.description,
            'latent_dim': self.latent_dim,
            'n_heads': self.n_heads,
            'head_dim': self.head_dim,
            'n_segments': self.n_segments,
            'encoder_hidden': self.encoder_hidden,
            'encoder_layers': self.encoder_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience,
            'grad_clip_norm': self.grad_clip_norm,
            'cluster_loss_weight': self.cluster_loss_weight,
            'separation_loss_weight': self.separation_loss_weight,
            'diversity_loss_weight': self.diversity_loss_weight,
            'use_trajectory_prototypes': self.use_trajectory_prototypes,
            'segment_weighting': self.segment_weighting.value,
            'n_folds': self.n_folds,
            'is_default': self.is_default,
        }


def get_default_config() -> AblationConfig:
    """Get the default MHTPN configuration (baseline for comparisons)."""
    return AblationConfig(
        ablation_id="DEFAULT",
        ablation_name="default",
        variant_name="MHTPN (Default)",
        description="Default MHTPN configuration with n_segments=8, trajectory prototypes, full loss, padding-aware weighting",
        is_default=True,
    )


def get_ablation1_configs() -> List[AblationConfig]:
    """
    Ablation 1: Number of Temporal Segments

    Tests n_segments = {1, 4, 8, 16}
    n_segments=8 is the default.

    Key hypothesis: n_segments=1 should approximate MultiHeadProtoNet (~70.9%)
    """
    configs = []

    # A1.1: n_segments=1 (no temporal segmentation)
    configs.append(AblationConfig(
        ablation_id="A1.1",
        ablation_name="n_segments",
        variant_name="n_segments=1",
        description="No temporal segmentation - approximates MultiHeadProtoNet",
        n_segments=1,
    ))

    # A1.2: n_segments=4 (coarse temporal)
    configs.append(AblationConfig(
        ablation_id="A1.2",
        ablation_name="n_segments",
        variant_name="n_segments=4",
        description="Coarse temporal segmentation",
        n_segments=4,
    ))

    # A1.3: n_segments=8 (default)
    configs.append(AblationConfig(
        ablation_id="A1.3",
        ablation_name="n_segments",
        variant_name="n_segments=8",
        description="Default temporal segmentation",
        n_segments=8,
        is_default=True,
    ))

    # A1.4: n_segments=16 (fine temporal)
    configs.append(AblationConfig(
        ablation_id="A1.4",
        ablation_name="n_segments",
        variant_name="n_segments=16",
        description="Fine temporal segmentation",
        n_segments=16,
    ))

    return configs


def get_ablation2_configs() -> List[AblationConfig]:
    """
    Ablation 2: Trajectory vs Static Prototypes

    Tests whether trajectory prototypes (origin + velocity)
    outperform static prototypes (velocity=0).
    """
    configs = []

    # A2.1: Trajectory prototypes (default)
    configs.append(AblationConfig(
        ablation_id="A2.1",
        ablation_name="trajectory",
        variant_name="Trajectory",
        description="Trajectory prototypes: p(t) = origin + t * velocity (default)",
        use_trajectory_prototypes=True,
        is_default=True,
    ))

    # A2.2: Static prototypes
    configs.append(AblationConfig(
        ablation_id="A2.2",
        ablation_name="trajectory",
        variant_name="Static",
        description="Static prototypes: velocity frozen to 0",
        use_trajectory_prototypes=False,
    ))

    return configs


def get_ablation3_configs() -> List[AblationConfig]:
    """
    Ablation 3: Loss Component Combinations

    Tests which loss components are essential:
    - CE only
    - CE + Cluster
    - CE + Separation
    - CE + Cluster + Separation
    - Full (CE + Cluster + Separation + Diversity) - default

    Expected: Cluster loss essential (+2%), Separation alone harmful (-5%)
    based on prior MultiHeadProtoNet results.
    """
    configs = []

    # A3.1: CE only
    configs.append(AblationConfig(
        ablation_id="A3.1",
        ablation_name="loss",
        variant_name="CE only",
        description="Cross-entropy loss only",
        cluster_loss_weight=0.0,
        separation_loss_weight=0.0,
        diversity_loss_weight=0.0,
    ))

    # A3.2: CE + Cluster
    configs.append(AblationConfig(
        ablation_id="A3.2",
        ablation_name="loss",
        variant_name="CE + Cluster",
        description="CE + Cluster loss (no separation, no diversity)",
        cluster_loss_weight=0.3,
        separation_loss_weight=0.0,
        diversity_loss_weight=0.0,
    ))

    # A3.3: CE + Separation
    configs.append(AblationConfig(
        ablation_id="A3.3",
        ablation_name="loss",
        variant_name="CE + Separation",
        description="CE + Separation loss (no cluster, no diversity)",
        cluster_loss_weight=0.0,
        separation_loss_weight=0.1,
        diversity_loss_weight=0.0,
    ))

    # A3.4: CE + Cluster + Separation
    configs.append(AblationConfig(
        ablation_id="A3.4",
        ablation_name="loss",
        variant_name="CE + Cl + Sep",
        description="CE + Cluster + Separation (no diversity)",
        cluster_loss_weight=0.3,
        separation_loss_weight=0.1,
        diversity_loss_weight=0.0,
    ))

    # A3.5: Full (default)
    configs.append(AblationConfig(
        ablation_id="A3.5",
        ablation_name="loss",
        variant_name="Full",
        description="Full loss: CE + Cluster + Separation + Diversity (default)",
        cluster_loss_weight=0.3,
        separation_loss_weight=0.1,
        diversity_loss_weight=0.05,
        is_default=True,
    ))

    return configs


def get_ablation4_configs() -> List[AblationConfig]:
    """
    Ablation 4: Segment Weighting Strategy

    Tests how segments should be weighted:
    - Uniform: All segments equal weight
    - Padding-aware: Weight by fraction of real data (default)
    - Learned attention: Trainable attention weights
    """
    configs = []

    # A4.1: Uniform weighting
    configs.append(AblationConfig(
        ablation_id="A4.1",
        ablation_name="weighting",
        variant_name="Uniform",
        description="All segments weighted equally (ignores padding)",
        segment_weighting=SegmentWeightingStrategy.UNIFORM,
    ))

    # A4.2: Padding-aware weighting (default)
    configs.append(AblationConfig(
        ablation_id="A4.2",
        ablation_name="weighting",
        variant_name="Padding-aware",
        description="Weight by fraction of real data (default)",
        segment_weighting=SegmentWeightingStrategy.PADDING_AWARE,
        is_default=True,
    ))

    # A4.3: Learned attention
    configs.append(AblationConfig(
        ablation_id="A4.3",
        ablation_name="weighting",
        variant_name="Learned",
        description="Trainable attention weights",
        segment_weighting=SegmentWeightingStrategy.LEARNED_ATTENTION,
    ))

    return configs


def get_all_ablation_configs() -> Dict[str, List[AblationConfig]]:
    """
    Get all ablation configurations organized by ablation study.

    Returns:
        Dict mapping ablation name to list of variants
    """
    return {
        'n_segments': get_ablation1_configs(),
        'trajectory': get_ablation2_configs(),
        'loss': get_ablation3_configs(),
        'weighting': get_ablation4_configs(),
    }


def count_total_runs() -> int:
    """Count total number of training runs (variants * folds)."""
    all_configs = get_all_ablation_configs()
    n_variants = sum(len(configs) for configs in all_configs.values())
    n_folds = 5
    return n_variants * n_folds


# Print summary when module is loaded
if __name__ == '__main__':
    all_configs = get_all_ablation_configs()
    total_variants = sum(len(configs) for configs in all_configs.values())

    print("MHTPN Ablation Study Configuration Summary")
    print("=" * 60)

    for ablation_name, configs in all_configs.items():
        print(f"\nAblation: {ablation_name} ({len(configs)} variants)")
        for config in configs:
            default_marker = " (default)" if config.is_default else ""
            print(f"  - {config.ablation_id}: {config.variant_name}{default_marker}")

    print(f"\n{'='*60}")
    print(f"Total variants: {total_variants}")
    print(f"Total training runs (5 folds): {total_variants * 5}")
