"""
Metrics Computation Module for SOTA Comparison

Computes all classification performance metrics with proper handling
of statistical measures (mean, std, confidence intervals).

Supports both binary classification (MG, Heartbeat) and multi-class
classification (BasicMotions, Epilepsy).
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
    sensitivity: float  # Recall for positive class (binary) or macro-avg (multi-class)
    specificity: float  # TNR for binary, macro-avg per-class specificity for multi-class
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    n_classes: int = 2

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (excluding confusion matrix)."""
        return {
            'accuracy': self.accuracy,
            'balanced_accuracy': self.balanced_accuracy,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'n_classes': self.n_classes,
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


def compute_multiclass_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged specificity for multi-class classification.

    Specificity for class i = TN_i / (TN_i + FP_i)
    where TN_i = samples correctly predicted as NOT class i
    and FP_i = samples incorrectly predicted as class i

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Macro-averaged specificity
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    specificities = []
    for cls in classes:
        # True negatives: correctly identified as NOT this class
        tn = np.sum((y_true != cls) & (y_pred != cls))
        # False positives: incorrectly predicted as this class
        fp = np.sum((y_true != cls) & (y_pred == cls))

        if (tn + fp) > 0:
            specificities.append(tn / (tn + fp))
        else:
            specificities.append(0.0)

    return np.mean(specificities)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> ClassificationMetrics:
    """
    Compute all classification metrics.

    Handles both binary (2 classes) and multi-class (>2 classes) classification.
    For multi-class:
    - Sensitivity = macro-averaged recall
    - Specificity = macro-averaged per-class specificity
    - F1 = macro-averaged F1
    - AUC = macro-averaged one-vs-rest AUC

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        ClassificationMetrics object
    """
    # Determine number of classes
    n_classes = len(np.unique(y_true))
    is_binary = (n_classes == 2)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Compute sensitivity and specificity
    if is_binary:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Multi-class: use macro average
        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
        specificity = compute_multiclass_specificity(y_true, y_pred)

    # Compute F1
    if is_binary:
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    else:
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Compute AUC-ROC
    try:
        if is_binary:
            # Use probability of positive class
            if y_proba.ndim == 2:
                proba_for_auc = y_proba[:, 1]
            else:
                proba_for_auc = y_proba
            auc = roc_auc_score(y_true, proba_for_auc)
        else:
            # Multi-class: use one-vs-rest with macro average
            if y_proba.ndim == 2 and y_proba.shape[1] == n_classes:
                auc = roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class='ovr',
                    average='macro',
                )
            else:
                # Fallback if probabilities are not in expected format
                auc = 0.5
    except ValueError:
        # Handle cases with single class or other edge cases
        auc = 0.5

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        sensitivity=sensitivity,
        specificity=specificity,
        f1_score=f1,
        auc_roc=auc,
        confusion_matrix=cm,
        n_classes=n_classes,
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
