"""
CCECE Paper: Time Series Models for MG Classification

Model Registry - Add new models here to make them available in experiments.
"""

from .base import BaseTimeSeriesModel, Attention
from .bigru_attention import BiGRUAttention
from .tcn import TCN
from .transformer import TransformerClassifier
from .lstm import VanillaLSTM, BiLSTMAttention
from .cnn1d import CNN1D
from .inceptiontime import InceptionTime
from .resnet1d import ResNet1D
from .concept_bottleneck_tcn import ConceptBottleneckTCN
from .temporal_concept_dynamics import TemporalConceptDynamicsNetwork
from .tcdn_enhanced import EnhancedTCDN
from .tcdn_faithful import TCDNFaithful
from .tcdn_gisa import TCDNFaithfulBottleneck
from .temp_proto_net import TempProtoNet

# Model Registry
# Add new models here as: 'name': ModelClass
MODEL_REGISTRY = {
    'bigru_attention': BiGRUAttention,
    'tcn': TCN,
    'transformer': TransformerClassifier,
    'lstm': VanillaLSTM,
    'bilstm_attention': BiLSTMAttention,
    'cnn1d': CNN1D,
    'inceptiontime': InceptionTime,
    'resnet1d': ResNet1D,
    'concept_tcn': ConceptBottleneckTCN,
    'tcdn': TemporalConceptDynamicsNetwork,
    'tcdn_enhanced': EnhancedTCDN,
    'tcdn_faithful': TCDNFaithful,
    'tcdn_faithful_bottleneck': TCDNFaithfulBottleneck,
    'temp_proto_net': TempProtoNet,
}


def get_model(name: str, **kwargs):
    """
    Get a model instance by name.

    Args:
        name: Model name (must be in MODEL_REGISTRY)
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    'BaseTimeSeriesModel',
    'Attention',
    'BiGRUAttention',
    'TCN',
    'TransformerClassifier',
    'VanillaLSTM',
    'BiLSTMAttention',
    'CNN1D',
    'InceptionTime',
    'ResNet1D',
    'ConceptBottleneckTCN',
    'TemporalConceptDynamicsNetwork',
    'EnhancedTCDN',
    'TCDNFaithful',
    'TCDNFaithfulBottleneck',
    'TempProtoNet',
    'MODEL_REGISTRY',
    'get_model',
    'list_models',
]
