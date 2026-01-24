"""
Ablation Study Configuration System

Defines all ablation variants while keeping other parameters at default values.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum


class FusionStrategy(Enum):
    """Fusion strategies for combining multi-head outputs."""
    AVERAGE = "average"
    MAX = "max"
    ATTENTION = "attention"
    VOTING = "voting"


class ClassificationType(Enum):
    """Classification head types."""
    PROTOTYPE = "prototype"
    FC = "fc"
    FC_DROPOUT = "fc_dropout"


@dataclass
class AblationConfig:
    """
    Configuration for a single ablation variant.

    Defaults match the successful MultiHeadProtoNet configuration from EXPERIMENT 01.
    """
    # Variant identification
    ablation_name: str = "default"
    variant_name: str = "default"
    is_default: bool = False

    # Model hyperparameters
    latent_dim: int = 64
    n_heads: int = 5
    head_dim: int = 64  # Increased from 32 based on EXPERIMENT 01
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2

    # Classification type
    classification_type: ClassificationType = ClassificationType.PROTOTYPE
    fc_dropout: float = 0.3  # Only used for FC_DROPOUT

    # Fusion strategy
    fusion_strategy: FusionStrategy = FusionStrategy.AVERAGE

    # Loss weights
    use_ce_loss: bool = True
    use_cluster_loss: bool = True
    use_separation_loss: bool = True
    cluster_loss_weight: float = 0.3
    separation_loss_weight: float = 0.1
    per_head_ce_weight: float = 0.0  # Disabled as per EXPERIMENT 01

    # Training hyperparameters (fixed across all ablations)
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    grad_clip_norm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum values converted to strings."""
        d = asdict(self)
        d['classification_type'] = self.classification_type.value
        d['fusion_strategy'] = self.fusion_strategy.value
        return d

    def get_loss_description(self) -> str:
        """Get a string description of the loss configuration."""
        components = []
        if self.use_ce_loss:
            components.append("CE")
        if self.use_cluster_loss:
            components.append("Cluster")
        if self.use_separation_loss:
            components.append("Sep")
        return "+".join(components) if components else "None"


# =============================================================================
# ABLATION 1: Number of Heads
# =============================================================================

def get_n_heads_variants() -> List[AblationConfig]:
    """
    Ablation 1: Vary the number of prototype heads.

    Question: Is multi-head architecture necessary? What is the optimal count?

    Variants: n_heads ∈ {1, 3, 5 (default), 7, 10}
    """
    variants = []

    for n_heads in [1, 3, 5, 7, 10]:
        is_default = (n_heads == 5)
        config = AblationConfig(
            ablation_name="n_heads",
            variant_name=f"n_heads_{n_heads}",
            is_default=is_default,
            n_heads=n_heads,
        )
        variants.append(config)

    return variants


# =============================================================================
# ABLATION 2: Prototype vs FC Classification
# =============================================================================

def get_classification_variants() -> List[AblationConfig]:
    """
    Ablation 2: Compare prototype-based vs standard FC classification.

    Question: Does prototype-based classification provide benefit?

    Variants:
    - Prototype (default): Distance to learned prototypes
    - FC: Same encoder + Linear(d_model, num_classes)
    - FC + Dropout: Same encoder + Dropout(0.3) + Linear
    """
    variants = []

    # Prototype (default)
    variants.append(AblationConfig(
        ablation_name="classification",
        variant_name="prototype",
        is_default=True,
        classification_type=ClassificationType.PROTOTYPE,
    ))

    # FC Classification
    variants.append(AblationConfig(
        ablation_name="classification",
        variant_name="fc",
        is_default=False,
        classification_type=ClassificationType.FC,
    ))

    # FC + Dropout
    variants.append(AblationConfig(
        ablation_name="classification",
        variant_name="fc_dropout",
        is_default=False,
        classification_type=ClassificationType.FC_DROPOUT,
        fc_dropout=0.3,
    ))

    return variants


# =============================================================================
# ABLATION 3: Loss Component Analysis
# =============================================================================

def get_loss_variants() -> List[AblationConfig]:
    """
    Ablation 3: Analyze contribution of each loss component.

    Question: Which loss components are essential for performance?

    Variants:
    - CE only (standard classification)
    - CE + Cluster (pull samples toward correct prototype)
    - CE + Separation (push incorrect prototypes apart)
    - Full (CE + Cluster + Separation) - default
    """
    variants = []

    # CE only
    variants.append(AblationConfig(
        ablation_name="loss",
        variant_name="ce_only",
        is_default=False,
        use_ce_loss=True,
        use_cluster_loss=False,
        use_separation_loss=False,
    ))

    # CE + Cluster
    variants.append(AblationConfig(
        ablation_name="loss",
        variant_name="ce_cluster",
        is_default=False,
        use_ce_loss=True,
        use_cluster_loss=True,
        use_separation_loss=False,
    ))

    # CE + Separation
    variants.append(AblationConfig(
        ablation_name="loss",
        variant_name="ce_separation",
        is_default=False,
        use_ce_loss=True,
        use_cluster_loss=False,
        use_separation_loss=True,
    ))

    # Full (default)
    variants.append(AblationConfig(
        ablation_name="loss",
        variant_name="full",
        is_default=True,
        use_ce_loss=True,
        use_cluster_loss=True,
        use_separation_loss=True,
    ))

    return variants


# =============================================================================
# ABLATION 4: Fusion Strategy
# =============================================================================

def get_fusion_variants() -> List[AblationConfig]:
    """
    Ablation 4: Compare different ways to combine multi-head outputs.

    Question: What is the best way to fuse predictions from multiple heads?

    Variants:
    - Average (default): Mean of head logits
    - Max: Max of head logits
    - Attention: Learned attention weights over heads
    - Voting: Majority vote of head predictions
    """
    variants = []

    # Average (default)
    variants.append(AblationConfig(
        ablation_name="fusion",
        variant_name="average",
        is_default=True,
        fusion_strategy=FusionStrategy.AVERAGE,
    ))

    # Max
    variants.append(AblationConfig(
        ablation_name="fusion",
        variant_name="max",
        is_default=False,
        fusion_strategy=FusionStrategy.MAX,
    ))

    # Attention
    variants.append(AblationConfig(
        ablation_name="fusion",
        variant_name="attention",
        is_default=False,
        fusion_strategy=FusionStrategy.ATTENTION,
    ))

    # Voting
    variants.append(AblationConfig(
        ablation_name="fusion",
        variant_name="voting",
        is_default=False,
        fusion_strategy=FusionStrategy.VOTING,
    ))

    return variants


# =============================================================================
# ABLATION 5: Encoder Architecture
# =============================================================================

def get_encoder_variants() -> List[AblationConfig]:
    """
    Ablation 5: Analyze sensitivity to encoder architecture.

    Question: How sensitive is performance to encoder depth/width?

    Variants:
    - 2 layers, 64 hidden (shallow, narrow)
    - 3 layers, 128 hidden (medium)
    - 3 layers, 64 hidden (default - note: original CONTEXT.md says 4 layers,
      but actual model uses 3 layers based on code)
    - 3 layers, 256 hidden (wider)
    - 4 layers, 64 hidden (deeper)

    Note: Based on multi_head_proto_net.py, the default is actually 3 layers
    with encoder_hidden=64. The tracker said 4 layers, 128 dim but this doesn't
    match the code. Using actual code values.
    """
    variants = []

    # Shallow, narrow
    variants.append(AblationConfig(
        ablation_name="encoder",
        variant_name="2L_64H",
        is_default=False,
        encoder_layers=2,
        encoder_hidden=64,
    ))

    # Medium
    variants.append(AblationConfig(
        ablation_name="encoder",
        variant_name="3L_128H",
        is_default=False,
        encoder_layers=3,
        encoder_hidden=128,
    ))

    # Default (3 layers, 64 hidden)
    variants.append(AblationConfig(
        ablation_name="encoder",
        variant_name="3L_64H",
        is_default=True,
        encoder_layers=3,
        encoder_hidden=64,
    ))

    # Wider
    variants.append(AblationConfig(
        ablation_name="encoder",
        variant_name="3L_256H",
        is_default=False,
        encoder_layers=3,
        encoder_hidden=256,
    ))

    # Deeper
    variants.append(AblationConfig(
        ablation_name="encoder",
        variant_name="4L_64H",
        is_default=False,
        encoder_layers=4,
        encoder_hidden=64,
    ))

    return variants


# =============================================================================
# Get All Variants
# =============================================================================

def get_all_ablation_variants() -> Dict[str, List[AblationConfig]]:
    """
    Get all ablation variants organized by ablation type.

    Returns:
        Dict mapping ablation name to list of variants
    """
    return {
        "n_heads": get_n_heads_variants(),
        "classification": get_classification_variants(),
        "loss": get_loss_variants(),
        "fusion": get_fusion_variants(),
        "encoder": get_encoder_variants(),
    }


def count_total_variants() -> int:
    """Count total number of unique variants (removing duplicates)."""
    all_variants = get_all_ablation_variants()

    # Count unique variants (default appears in each ablation)
    total = 0
    seen_default = False

    for ablation_name, variants in all_variants.items():
        for v in variants:
            if v.is_default:
                if not seen_default:
                    total += 1
                    seen_default = True
            else:
                total += 1

    return total


def get_default_config() -> AblationConfig:
    """Get the default (baseline) configuration."""
    return AblationConfig(
        ablation_name="default",
        variant_name="default",
        is_default=True,
    )


# Print summary when run directly
if __name__ == "__main__":
    all_variants = get_all_ablation_variants()

    print("=" * 60)
    print("ABLATION STUDY VARIANTS")
    print("=" * 60)

    total = 0
    for ablation_name, variants in all_variants.items():
        print(f"\n{ablation_name.upper()}:")
        for v in variants:
            marker = " (DEFAULT)" if v.is_default else ""
            print(f"  - {v.variant_name}{marker}")
            total += 1

    print(f"\nTotal variants: {total}")
    print(f"Unique variants (counting default once): {count_total_variants()}")
    print(f"Training runs (variants × 5 folds): {count_total_variants() * 5}")
