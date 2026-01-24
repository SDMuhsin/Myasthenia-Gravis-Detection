"""
CCECE Paper: Experiment 03 - Explainability Analysis

Provides quantitative and visual explainability for MultiHeadProtoNet:
1. Prototype Analysis - Are prototypes meaningful?
2. Temporal Saliency - Which time regions matter?
3. Feature Importance - Which channels drive predictions?
4. Head Specialization - Do heads specialize differently?
5. Case Studies - What distinguishes correct/incorrect predictions?
"""

from .prototype_analysis import PrototypeAnalyzer
from .saliency import TemporalSaliencyAnalyzer
from .feature_importance import FeatureImportanceAnalyzer
from .head_analysis import HeadSpecializationAnalyzer
from .case_studies import CaseStudyAnalyzer
from .report_generator import ReportGenerator

__all__ = [
    'PrototypeAnalyzer',
    'TemporalSaliencyAnalyzer',
    'FeatureImportanceAnalyzer',
    'HeadSpecializationAnalyzer',
    'CaseStudyAnalyzer',
    'ReportGenerator',
]
