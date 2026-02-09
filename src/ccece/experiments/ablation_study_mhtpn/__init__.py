"""
MHTPN Ablation Study Package

Contains ablation configurations, runner, and analysis tools for
validating MultiHeadTrajectoryProtoNet architectural choices.
"""

from .configs import AblationConfig, get_all_ablation_configs
from .runner import run_ablation_study
from .analysis import analyze_results, statistical_tests
from .figure_generator import generate_all_figures

__all__ = [
    'AblationConfig',
    'get_all_ablation_configs',
    'run_ablation_study',
    'analyze_results',
    'statistical_tests',
    'generate_all_figures',
]
