"""
SOTA Comparison Experiment for CCECE Paper

This module contains implementations of state-of-the-art time series classification
methods for comparison with MultiHeadProtoNet.

Models:
    Simple Baselines:
    - SimpleCNN: 3-layer 1D CNN
    - SimpleLSTM: 2-layer BiLSTM

    SOTA Methods (2020-2025):
    - InceptionTime (DMKD 2020)
    - ROCKET (DMKD 2020)
    - TST (KDD 2021)
    - TimesNet (ICLR 2023)
    - ConvTran (DMKD 2023)
    - PatchTST (ICLR 2023)

    Our Method:
    - MultiHeadProtoNet
"""

from .runner import run_sota_comparison

__all__ = ['run_sota_comparison']
