"""
Metrics Computation Module for SOTA Comparison

Computes all classification performance metrics with proper handling
of statistical measures (mean, std, confidence intervals).
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics from a single evaluation."""
    accuracy: float
    balanced_accuracy: float
    sensitivity: float  # Recall for MG (positive class)
    specificity: float  # Recall for HC (negative class)
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (excluding confusion matrix)."""
        return {
            'accuracy': self.accuracy,
            'balanced_accuracy': self.balanced_accuracy,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
        }

    def __str__(self) -> str:
        return (
            f"Acc: {self.accuracy:.4f} | "
            f"BalAcc: {self.balanced_accuracy:.4f} | "
            f"Sens: {self.sensitivity:.4f} | "
            f"Spec: {self.specificity:.4f} | "
            f"F1: {self.f1_score:.4f} | "
            f"AUC: {self.auc_roc:.4f}"
        )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> ClassificationMetrics:
    """
    Compute all classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for positive class in binary)

    Returns:
        ClassificationMetrics object
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # For binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Multiclass - use macro average
        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
        specificity = 0.0  # Not well-defined for multiclass

    # Handle probability for AUC
    if y_proba.ndim == 2:
        # Use probability of positive class for binary
        proba_for_auc = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
    else:
        proba_for_auc = y_proba

    try:
        auc = roc_auc_score(y_true, proba_for_auc)
    except ValueError:
        auc = 0.5  # Default for cases with only one class

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        sensitivity=sensitivity,
        specificity=specificity,
        f1_score=f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'macro', zero_division=0),
        auc_roc=auc,
        confusion_matrix=cm,
    )


def aggregate_metrics(
    fold_metrics: List[ClassificationMetrics],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across folds with mean, std, and 95% CI.

    Args:
        fold_metrics: List of ClassificationMetrics from each fold

    Returns:
        Dict mapping metric name to {'mean', 'std', 'ci_lower', 'ci_upper'}
    """
    metric_names = ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc_roc']

    aggregated = {}
    for metric_name in metric_names:
        values = [getattr(m, metric_name) for m in fold_metrics]
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0

        # 95% CI using t-distribution
        n = len(values)
        if n > 1:
            from scipy import stats
            t_val = stats.t.ppf(0.975, n - 1)
            ci_margin = t_val * std / np.sqrt(n)
        else:
            ci_margin = 0.0

        aggregated[metric_name] = {
            'mean': mean,
            'std': std,
            'ci_lower': mean - ci_margin,
            'ci_upper': mean + ci_margin,
            'values': values,
        }

    return aggregated


def format_metric(mean: float, std: float, as_percent: bool = True) -> str:
    """Format metric as 'mean ± std' string."""
    if as_percent:
        return f"{mean*100:.1f} ± {std*100:.1f}"
    else:
        return f"{mean:.3f} ± {std:.3f}"
