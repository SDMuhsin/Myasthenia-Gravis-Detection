"""
Ablation Study for MultiHeadProtoNet

Systematically evaluates the contribution of each component:
1. Number of Heads (n_heads)
2. Prototype vs FC Classification
3. Loss Components
4. Fusion Strategy
5. Encoder Architecture
"""

from .configs import (
    AblationConfig,
    get_n_heads_variants,
    get_classification_variants,
    get_loss_variants,
    get_fusion_variants,
    get_encoder_variants,
    get_all_ablation_variants,
)
