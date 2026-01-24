"""
Baseline model implementations for SOTA comparison.
"""

from .base import BaselineModel
from .simple_cnn import SimpleCNN
from .simple_lstm import SimpleLSTM
from .inception_time import InceptionTimeWrapper
from .rocket import ROCKETWrapper
from .tst import TimeSeriesTransformer
from .timesnet import TimesNetWrapper
from .convtran import ConvTranWrapper
from .patchtst import PatchTSTWrapper

__all__ = [
    'BaselineModel',
    'SimpleCNN',
    'SimpleLSTM',
    'InceptionTimeWrapper',
    'ROCKETWrapper',
    'TimeSeriesTransformer',
    'TimesNetWrapper',
    'ConvTranWrapper',
    'PatchTSTWrapper',
]

# Registry of all baseline models
BASELINE_REGISTRY = {
    '1D-CNN': SimpleCNN,
    'LSTM': SimpleLSTM,
    'InceptionTime': InceptionTimeWrapper,
    'ROCKET': ROCKETWrapper,
    'TST': TimeSeriesTransformer,
    'TimesNet': TimesNetWrapper,
    'ConvTran': ConvTranWrapper,
    'PatchTST': PatchTSTWrapper,
}

def get_baseline(name: str, **kwargs):
    """Get a baseline model by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[name](**kwargs)

def list_baselines():
    """List available baseline models."""
    return list(BASELINE_REGISTRY.keys())
