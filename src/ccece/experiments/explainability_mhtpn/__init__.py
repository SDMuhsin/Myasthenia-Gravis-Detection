"""
MHTPN Explainability Analysis Module

Analyzes intrinsic explainability features of MultiHeadTrajectoryProtoNet:
1. Trajectory Prototype Analysis - learned origins and velocities
2. Per-Segment Decision Analysis - which segments drive classification
3. Head Velocity Diversity - different heads capture different dynamics
4. Sample-Level Case Studies - individual patient trajectory analysis
"""

from .trajectory_analysis import analyze_trajectory_prototypes
from .segment_analysis import analyze_per_segment_decisions
from .velocity_diversity import analyze_velocity_diversity
from .case_studies import analyze_case_studies
from .runner import run_explainability

__all__ = [
    'analyze_trajectory_prototypes',
    'analyze_per_segment_decisions',
    'analyze_velocity_diversity',
    'analyze_case_studies',
    'run_explainability',
]
