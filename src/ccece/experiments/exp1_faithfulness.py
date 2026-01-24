#!/usr/bin/env python3
"""
CCECE Paper: Experiment 1 - Faithfulness Benchmark

Quantitatively compare TCDN's intrinsic explanations against post-hoc methods
using established faithfulness metrics (Sufficiency, Comprehensiveness, AOPC).

Author: CCECE Experiment Agent
Date: 2026-01-17
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, compute_target_seq_len, set_all_seeds
from ccece.trainer import (
    TrainingConfig, Trainer, create_data_loaders, SequenceScaler,
    SaccadeDataset, EvaluationMetrics
)
from ccece.models import get_model

# Try to import captum for post-hoc methods
try:
    from captum.attr import IntegratedGradients, GradientShap, Saliency, NoiseTunnel
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("WARNING: captum not available. Install with: pip install captum")

# Try to import thop for FLOPs computation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("WARNING: thop not available. Install with: pip install thop")


# =============================================================================
# CONSTANTS
# =============================================================================

BASE_OUTPUT_DIR = './results/ccece/tcdn_experiments/exp1_faithfulness'
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Faithfulness evaluation percentiles
K_VALUES = [5, 10, 20, 50, 100]

# Methods to compare
METHODS = [
    'TCDN',
    'TCDNFaithful',
    'TCDNFaithful+IG',
    'IntegratedGradients',
    'GradientSHAP',
    'Saliency',
    'SmoothGrad',
    'Attention',
    'Random',
]


# =============================================================================
# EXPLANATION METHODS
# =============================================================================

class ExplanationMethod:
    """Base class for explanation methods."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def explain(self, x: torch.Tensor) -> np.ndarray:
        """
        Generate importance scores for input.

        Args:
            x: Input tensor (1, seq_len, input_dim)

        Returns:
            Importance scores (seq_len,) - higher = more important
        """
        raise NotImplementedError


class TCDNExplanation(ExplanationMethod):
    """
    TCDN intrinsic explanation based on concept trajectory contributions.

    Maps trajectory feature importance back to temporal segments, then broadcasts
    to input timesteps.
    """

    def explain(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device)
        batch_size, seq_len, input_dim = x.shape

        # Forward pass to get trajectory info
        with torch.no_grad():
            logits, traj_info = self.model.forward_with_trajectory(x)
            pred_class = logits.argmax(dim=1).item()

        # Get segment concepts: (1, num_segments, num_concepts)
        segment_concepts = traj_info['segment_concepts'][0].cpu().numpy()  # (num_segments, num_concepts)
        num_segments = segment_concepts.shape[0]

        # Compute segment importance based on concept variance and deviation
        # Higher variance in a segment = more informative
        # Segments that differ from mean = more discriminative
        segment_importance = np.zeros(num_segments)

        for seg_idx in range(num_segments):
            # Importance based on concept magnitude (higher concepts = more active)
            seg_concepts = segment_concepts[seg_idx]
            segment_importance[seg_idx] = np.abs(seg_concepts).mean()

        # Add trajectory-based importance (slope contribution)
        trajectory_features = traj_info['trajectory_features'][0].cpu().numpy()
        num_concepts = 5
        num_traj_features = 5

        # Extract slopes (index 2 for each concept)
        for c_idx in range(num_concepts):
            slope_idx = c_idx * num_traj_features + 2  # slope is 3rd feature
            slope = abs(trajectory_features[slope_idx])
            # Higher slope means later segments differ more from earlier
            for seg_idx in range(num_segments):
                # Weight increases with segment index for positive slope
                weight = (seg_idx + 1) / num_segments
                segment_importance[seg_idx] += slope * weight

        # Normalize segment importance
        segment_importance = segment_importance / (segment_importance.max() + 1e-8)

        # Broadcast to timesteps
        segment_len = seq_len // num_segments
        timestep_importance = np.zeros(seq_len)

        for seg_idx in range(num_segments):
            start = seg_idx * segment_len
            end = start + segment_len if seg_idx < num_segments - 1 else seq_len
            timestep_importance[start:end] = segment_importance[seg_idx]

        return timestep_importance


class TCDNFaithfulExplanation(ExplanationMethod):
    """
    TCDN-Faithful intrinsic explanation using per-timestep attention and gradients.

    Uses forward_with_explanation() to get sparse importance scores that combine
    attention weights and gradient-based attribution.
    """

    def explain(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device)

        # Use forward_with_explanation which computes gradients internally
        logits, explanation = self.model.forward_with_explanation(x)

        # Return sparse importance scores
        sparse_importance = explanation['sparse_importance'][0].cpu().numpy()

        return sparse_importance


class TCDNFaithfulIGExplanation(ExplanationMethod):
    """
    Hybrid: IntegratedGradients on TCDN-Faithful for maximum faithfulness.

    Combines:
    - IntegratedGradients for faithful per-timestep importance
    - TCDN's concept trajectories for clinical interpretability

    This gives us the best of both worlds: faithful attributions that
    identify which timesteps matter, plus interpretable clinical concepts.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        if CAPTUM_AVAILABLE:
            self.ig = IntegratedGradients(self._forward_func)

    def _forward_func(self, x):
        return self.model(x)

    def explain(self, x: torch.Tensor) -> np.ndarray:
        if not CAPTUM_AVAILABLE:
            return np.random.rand(x.shape[1])

        x = x.to(self.device)
        x.requires_grad = True

        with torch.no_grad():
            pred = self.model(x).argmax(dim=1)

        # Compute IntegratedGradients attributions
        baseline = torch.zeros_like(x)
        attr = self.ig.attribute(x, baseline, target=pred, n_steps=50)

        # Sum over input features, take absolute value for importance
        importance = attr[0].abs().sum(dim=1).detach().cpu().numpy()
        importance = importance / (importance.max() + 1e-8)

        return importance

    def explain_with_concepts(self, x: torch.Tensor) -> dict:
        """
        Get both faithful importance AND clinical concept analysis.

        Returns:
            dict with 'importance' (faithful IG scores) and 'concepts' (trajectory analysis)
        """
        # Get faithful importance from IG
        importance = self.explain(x)

        # Get concept trajectories from TCDN
        x = x.to(self.device)
        with torch.no_grad():
            logits, traj_info = self.model.forward_with_trajectory(x)

        return {
            'importance': importance,
            'segment_concepts': traj_info['segment_concepts'],
            'trajectory_features': traj_info['trajectory_features'],
        }


class IntegratedGradientsExplanation(ExplanationMethod):
    """Integrated Gradients from Captum."""

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        if CAPTUM_AVAILABLE:
            self.ig = IntegratedGradients(self._forward_func)

    def _forward_func(self, x):
        return self.model(x)

    def explain(self, x: torch.Tensor) -> np.ndarray:
        if not CAPTUM_AVAILABLE:
            return np.random.rand(x.shape[1])

        x = x.to(self.device)
        x.requires_grad = True

        with torch.no_grad():
            pred = self.model(x).argmax(dim=1)

        # Compute attributions
        baseline = torch.zeros_like(x)
        attr = self.ig.attribute(x, baseline, target=pred, n_steps=50)

        # Sum over input features, take absolute value for importance
        importance = attr[0].abs().sum(dim=1).detach().cpu().numpy()  # (seq_len,)
        importance = importance / (importance.max() + 1e-8)

        return importance


class GradientSHAPExplanation(ExplanationMethod):
    """GradientSHAP from Captum."""

    def __init__(self, model: nn.Module, device: torch.device, baseline_samples: torch.Tensor = None):
        super().__init__(model, device)
        if CAPTUM_AVAILABLE:
            self.gshap = GradientShap(self._forward_func)
        self.baseline_samples = baseline_samples

    def _forward_func(self, x):
        return self.model(x)

    def set_baselines(self, baseline_samples: torch.Tensor):
        """Set baseline samples for SHAP."""
        self.baseline_samples = baseline_samples

    def explain(self, x: torch.Tensor) -> np.ndarray:
        if not CAPTUM_AVAILABLE:
            return np.random.rand(x.shape[1])

        x = x.to(self.device)
        x.requires_grad = True

        with torch.no_grad():
            pred = self.model(x).argmax(dim=1)

        # Use baseline samples if available, otherwise use zeros
        if self.baseline_samples is not None:
            baselines = self.baseline_samples[:10].to(self.device)
        else:
            baselines = torch.zeros(1, *x.shape[1:]).to(self.device)

        attr = self.gshap.attribute(x, baselines, target=pred, n_samples=5)

        importance = attr[0].abs().sum(dim=1).detach().cpu().numpy()
        importance = importance / (importance.max() + 1e-8)

        return importance


class SaliencyExplanation(ExplanationMethod):
    """Vanilla Saliency Maps from Captum."""

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        if CAPTUM_AVAILABLE:
            self.saliency = Saliency(self._forward_func)

    def _forward_func(self, x):
        return self.model(x)

    def explain(self, x: torch.Tensor) -> np.ndarray:
        if not CAPTUM_AVAILABLE:
            return np.random.rand(x.shape[1])

        x = x.to(self.device)
        x.requires_grad = True

        with torch.no_grad():
            pred = self.model(x).argmax(dim=1)

        attr = self.saliency.attribute(x, target=pred)

        importance = attr[0].abs().sum(dim=1).detach().cpu().numpy()
        importance = importance / (importance.max() + 1e-8)

        return importance


class SmoothGradExplanation(ExplanationMethod):
    """SmoothGrad (Saliency with noise tunnel) from Captum."""

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        if CAPTUM_AVAILABLE:
            self.saliency = Saliency(self._forward_func)
            self.smooth = NoiseTunnel(self.saliency)

    def _forward_func(self, x):
        return self.model(x)

    def explain(self, x: torch.Tensor) -> np.ndarray:
        if not CAPTUM_AVAILABLE:
            return np.random.rand(x.shape[1])

        x = x.to(self.device)
        x.requires_grad = True

        with torch.no_grad():
            pred = self.model(x).argmax(dim=1)

        attr = self.smooth.attribute(x, nt_type='smoothgrad', nt_samples=10,
                                      stdevs=0.1, target=pred)

        importance = attr[0].abs().sum(dim=1).detach().cpu().numpy()
        importance = importance / (importance.max() + 1e-8)

        return importance


class AttentionExplanation(ExplanationMethod):
    """Attention weights from BiGRU+Attention model."""

    def explain(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device)

        # Get attention weights
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(x)

        if attention_weights is None:
            return np.random.rand(x.shape[1])

        importance = attention_weights[0].cpu().numpy()  # (seq_len,)
        importance = importance / (importance.max() + 1e-8)

        return importance


class RandomExplanation(ExplanationMethod):
    """Random baseline explanation."""

    def explain(self, x: torch.Tensor) -> np.ndarray:
        seq_len = x.shape[1]
        importance = np.random.rand(seq_len)
        importance = importance / (importance.max() + 1e-8)
        return importance


# =============================================================================
# FAITHFULNESS METRICS
# =============================================================================

def mask_top_k_percent(x: torch.Tensor, importance: np.ndarray, k: float,
                       keep: bool = True) -> torch.Tensor:
    """
    Mask input based on importance scores.

    Args:
        x: Input tensor (1, seq_len, input_dim)
        importance: Importance scores (seq_len,)
        k: Percentage of features to select
        keep: If True, keep top-k% (for sufficiency). If False, remove top-k% (for comprehensiveness).

    Returns:
        Masked input tensor
    """
    x = x.clone()
    seq_len = x.shape[1]
    num_select = max(1, int(seq_len * k / 100))

    # Get indices of top-k% important features
    top_indices = np.argsort(importance)[-num_select:]

    if keep:
        # For sufficiency: zero out everything except top-k%
        mask = np.zeros(seq_len, dtype=bool)
        mask[top_indices] = True
    else:
        # For comprehensiveness: zero out top-k%
        mask = np.ones(seq_len, dtype=bool)
        mask[top_indices] = False

    # Apply mask (zero out masked timesteps)
    for t in range(seq_len):
        if not mask[t]:
            x[0, t, :] = 0

    return x


def compute_sufficiency(model: nn.Module, x: torch.Tensor, importance: np.ndarray,
                       k_values: List[float], device: torch.device) -> Dict[float, float]:
    """
    Compute sufficiency: accuracy using only top-k% features.

    Higher sufficiency = explanation identifies features sufficient for prediction.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        x = x.to(device)
        original_pred = model(x).argmax(dim=1).item()
        original_prob = F.softmax(model(x), dim=1)[0, original_pred].item()

    for k in k_values:
        x_masked = mask_top_k_percent(x, importance, k, keep=True).to(device)

        with torch.no_grad():
            masked_output = model(x_masked)
            masked_pred = masked_output.argmax(dim=1).item()
            masked_prob = F.softmax(masked_output, dim=1)[0, original_pred].item()

        # Sufficiency = how much of original prediction is retained
        results[k] = masked_prob

    return results


def compute_comprehensiveness(model: nn.Module, x: torch.Tensor, importance: np.ndarray,
                              k_values: List[float], device: torch.device) -> Dict[float, float]:
    """
    Compute comprehensiveness: probability drop when removing top-k% features.

    Higher comprehensiveness = explanation identifies features that matter for prediction.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        x = x.to(device)
        original_output = model(x)
        original_pred = original_output.argmax(dim=1).item()
        original_prob = F.softmax(original_output, dim=1)[0, original_pred].item()

    for k in k_values[:-1]:  # Exclude 100% (would remove everything)
        x_masked = mask_top_k_percent(x, importance, k, keep=False).to(device)

        with torch.no_grad():
            masked_output = model(x_masked)
            masked_prob = F.softmax(masked_output, dim=1)[0, original_pred].item()

        # Comprehensiveness = drop in probability
        results[k] = original_prob - masked_prob

    return results


def compute_aopc(comprehensiveness_results: Dict[float, List[float]]) -> Dict[str, float]:
    """
    Compute Area Over Perturbation Curve (AOPC).

    AOPC = average comprehensiveness across all k values.
    """
    aopc_per_method = {}

    for method, results_per_k in comprehensiveness_results.items():
        # Average across all k values for this method
        all_values = []
        for k, values in results_per_k.items():
            all_values.extend(values)
        aopc_per_method[method] = np.mean(all_values) if all_values else 0.0

    return aopc_per_method


def compute_sparsity(importance: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute sparsity: fraction of near-zero importance values.

    Higher sparsity = more focused explanation.
    """
    near_zero = np.abs(importance) < threshold
    return np.mean(near_zero)


# =============================================================================
# COMPUTATIONAL METRICS
# =============================================================================

def compute_model_metrics(model: nn.Module, seq_len: int, input_dim: int,
                          device: torch.device) -> Dict[str, Any]:
    """Compute computational metrics for a model."""
    metrics = {}

    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    metrics['num_parameters'] = num_params

    # Model size on disk
    torch.save(model.state_dict(), '/tmp/model_temp.pt')
    metrics['model_size_mb'] = os.path.getsize('/tmp/model_temp.pt') / (1024 * 1024)
    os.remove('/tmp/model_temp.pt')

    # Inference latency
    model.eval()
    x = torch.randn(1, seq_len, input_dim).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    metrics['latency_mean_ms'] = np.mean(latencies)
    metrics['latency_std_ms'] = np.std(latencies)

    # GPU memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x)
        metrics['gpu_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        metrics['gpu_memory_mb'] = 0

    # FLOPs
    if THOP_AVAILABLE:
        try:
            model_copy = model.cpu()
            x_cpu = x.cpu()
            flops, params = profile(model_copy, inputs=(x_cpu,), verbose=False)
            metrics['flops'] = flops
            model.to(device)
        except Exception as e:
            metrics['flops'] = None
            metrics['flops_error'] = str(e)
    else:
        metrics['flops'] = None

    return metrics


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the complete faithfulness benchmark experiment."""

    print("=" * 70)
    print("EXPERIMENT 1: FAITHFULNESS BENCHMARK")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print()

    # Set seed for reproducibility
    set_all_seeds(RANDOM_SEED)

    # Track total experiment time
    experiment_start_time = time.time()

    # ==========================================================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # ==========================================================================
    print("\n[STEP 1/6] Loading and preprocessing data...")

    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)
    X, y, patient_ids = extract_arrays(items)

    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]

    print(f"  Total samples: {len(items)}")
    print(f"  HC: {np.sum(y == 0)}, MG: {np.sum(y == 1)}")
    print(f"  Sequence length: {seq_len}, Features: {input_dim}")

    # Use 80/20 split for train/test
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

    train_items = [items[i] for i in train_idx]
    test_items = [items[i] for i in test_idx]
    train_labels = y[train_idx]
    test_labels = y[test_idx]

    print(f"  Train samples: {len(train_items)}")
    print(f"  Test samples: {len(test_items)}")

    # Create scaler from training data
    scaler = SequenceScaler().fit(train_items)

    # Create test dataset
    test_dataset = SaccadeDataset(test_items, seq_len, scaler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ==========================================================================
    # STEP 2: TRAIN MODELS
    # ==========================================================================
    print("\n[STEP 2/6] Training models...")

    models = {}
    model_metrics = {}

    # Training config (reduced epochs for faster experimentation)
    config = TrainingConfig(
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        early_stopping_patience=10,
    )

    # Create data loaders for training
    train_loader, val_loader, _ = create_data_loaders(
        train_items, test_items, seq_len, config.batch_size, scaler
    )

    # --- Train TCDN ---
    print("  Training TCDN...")
    tcdn_model = get_model('tcdn', input_dim=input_dim, num_classes=2, seq_len=seq_len)
    trainer = Trainer(tcdn_model, config, DEVICE)
    history = trainer.train(train_loader, val_loader, train_labels, verbose=False)
    models['TCDN'] = tcdn_model.to(DEVICE)
    model_metrics['TCDN'] = compute_model_metrics(tcdn_model, seq_len, input_dim, DEVICE)

    # Get TCDN accuracy
    metrics = trainer.evaluate(val_loader)
    print(f"    TCDN accuracy: {metrics.accuracy:.4f}")
    model_metrics['TCDN']['accuracy'] = metrics.accuracy
    model_metrics['TCDN']['auc'] = metrics.auc_roc
    model_metrics['TCDN']['sensitivity'] = metrics.sensitivity
    model_metrics['TCDN']['specificity'] = metrics.specificity

    # --- Train TCDN-Faithful ---
    print("  Training TCDN-Faithful...")
    tcdn_faithful_model = get_model('tcdn_faithful', input_dim=input_dim, num_classes=2, seq_len=seq_len)
    trainer = Trainer(tcdn_faithful_model, config, DEVICE)
    history = trainer.train(train_loader, val_loader, train_labels, verbose=False)
    models['TCDNFaithful'] = tcdn_faithful_model.to(DEVICE)
    model_metrics['TCDNFaithful'] = compute_model_metrics(tcdn_faithful_model, seq_len, input_dim, DEVICE)

    # Get TCDN-Faithful accuracy
    metrics = trainer.evaluate(val_loader)
    print(f"    TCDN-Faithful accuracy: {metrics.accuracy:.4f}")
    model_metrics['TCDNFaithful']['accuracy'] = metrics.accuracy
    model_metrics['TCDNFaithful']['auc'] = metrics.auc_roc
    model_metrics['TCDNFaithful']['sensitivity'] = metrics.sensitivity
    model_metrics['TCDNFaithful']['specificity'] = metrics.specificity

    # --- Train TCN (base model for post-hoc methods) ---
    print("  Training TCN (for post-hoc methods)...")
    tcn_model = get_model('tcn', input_dim=input_dim, num_classes=2, seq_len=seq_len)
    trainer = Trainer(tcn_model, config, DEVICE)
    history = trainer.train(train_loader, val_loader, train_labels, verbose=False)
    models['TCN'] = tcn_model.to(DEVICE)
    model_metrics['TCN'] = compute_model_metrics(tcn_model, seq_len, input_dim, DEVICE)

    metrics = trainer.evaluate(val_loader)
    print(f"    TCN accuracy: {metrics.accuracy:.4f}")
    model_metrics['TCN']['accuracy'] = metrics.accuracy
    model_metrics['TCN']['auc'] = metrics.auc_roc

    # --- Train BiGRU+Attention (for attention baseline) ---
    print("  Training BiGRU+Attention...")
    attention_model = get_model('bigru_attention', input_dim=input_dim, num_classes=2, seq_len=seq_len)
    trainer = Trainer(attention_model, config, DEVICE)
    history = trainer.train(train_loader, val_loader, train_labels, verbose=False)
    models['Attention'] = attention_model.to(DEVICE)
    model_metrics['Attention'] = compute_model_metrics(attention_model, seq_len, input_dim, DEVICE)

    metrics = trainer.evaluate(val_loader)
    print(f"    BiGRU+Attention accuracy: {metrics.accuracy:.4f}")
    model_metrics['Attention']['accuracy'] = metrics.accuracy
    model_metrics['Attention']['auc'] = metrics.auc_roc

    # ==========================================================================
    # STEP 3: CREATE EXPLANATION METHODS
    # ==========================================================================
    print("\n[STEP 3/6] Creating explanation methods...")

    # Get some baseline samples for GradientSHAP
    baseline_samples = []
    for x, _ in train_loader:
        baseline_samples.append(x)
        if len(baseline_samples) >= 3:
            break
    baseline_samples = torch.cat(baseline_samples, dim=0)[:20]

    explainers = {
        'TCDN': TCDNExplanation(models['TCDN'], DEVICE),
        'TCDNFaithful': TCDNFaithfulExplanation(models['TCDNFaithful'], DEVICE),
        'TCDNFaithful+IG': TCDNFaithfulIGExplanation(models['TCDNFaithful'], DEVICE),
        'IntegratedGradients': IntegratedGradientsExplanation(models['TCN'], DEVICE),
        'GradientSHAP': GradientSHAPExplanation(models['TCN'], DEVICE, baseline_samples),
        'Saliency': SaliencyExplanation(models['TCN'], DEVICE),
        'SmoothGrad': SmoothGradExplanation(models['TCN'], DEVICE),
        'Attention': AttentionExplanation(models['Attention'], DEVICE),
        'Random': RandomExplanation(models['TCN'], DEVICE),
    }

    print(f"  Created {len(explainers)} explanation methods")

    # ==========================================================================
    # STEP 4: COMPUTE FAITHFULNESS METRICS
    # ==========================================================================
    print("\n[STEP 4/6] Computing faithfulness metrics...")
    print(f"  Evaluating on {len(test_items)} test samples")
    print(f"  K values: {K_VALUES}")

    # Storage for results
    sufficiency_results = {method: {k: [] for k in K_VALUES} for method in METHODS}
    comprehensiveness_results = {method: {k: [] for k in K_VALUES[:-1]} for method in METHODS}
    sparsity_results = {method: [] for method in METHODS}
    explanation_times = {method: [] for method in METHODS}

    # Evaluate on test set
    for idx, (x, label) in enumerate(tqdm(test_loader, desc="  Evaluating samples")):
        x = x.to(DEVICE)

        for method in METHODS:
            # Choose the right model for evaluation
            if method == 'TCDN':
                eval_model = models['TCDN']
            elif method in ['TCDNFaithful', 'TCDNFaithful+IG']:
                eval_model = models['TCDNFaithful']
            elif method == 'Attention':
                eval_model = models['Attention']
            else:
                eval_model = models['TCN']

            # Time the explanation
            start_time = time.perf_counter()
            importance = explainers[method].explain(x)
            explanation_times[method].append(time.perf_counter() - start_time)

            # Compute sparsity
            sparsity_results[method].append(compute_sparsity(importance))

            # Compute sufficiency
            suff = compute_sufficiency(eval_model, x, importance, K_VALUES, DEVICE)
            for k, val in suff.items():
                sufficiency_results[method][k].append(val)

            # Compute comprehensiveness
            comp = compute_comprehensiveness(eval_model, x, importance, K_VALUES, DEVICE)
            for k, val in comp.items():
                comprehensiveness_results[method][k].append(val)

    # ==========================================================================
    # STEP 5: AGGREGATE AND SAVE RESULTS
    # ==========================================================================
    print("\n[STEP 5/6] Aggregating and saving results...")

    # --- Table 1: Sufficiency ---
    sufficiency_df_data = []
    for method in METHODS:
        row = {'Method': method}
        for k in K_VALUES:
            values = sufficiency_results[method][k]
            row[f'Top-{k}%'] = f"{np.mean(values):.4f}"
            row[f'Top-{k}%_std'] = f"{np.std(values):.4f}"
        sufficiency_df_data.append(row)

    sufficiency_df = pd.DataFrame(sufficiency_df_data)
    sufficiency_path = os.path.join(BASE_OUTPUT_DIR, 'results', 'exp1_table1_sufficiency.csv')
    sufficiency_df.to_csv(sufficiency_path, index=False)
    print(f"  Saved: {sufficiency_path}")

    # --- Table 2: Comprehensiveness ---
    comprehensiveness_df_data = []
    for method in METHODS:
        row = {'Method': method}
        for k in K_VALUES[:-1]:  # Exclude 100%
            values = comprehensiveness_results[method][k]
            row[f'Rm-{k}%'] = f"{np.mean(values):.4f}"
            row[f'Rm-{k}%_std'] = f"{np.std(values):.4f}"
        comprehensiveness_df_data.append(row)

    comprehensiveness_df = pd.DataFrame(comprehensiveness_df_data)
    comprehensiveness_path = os.path.join(BASE_OUTPUT_DIR, 'results', 'exp1_table2_comprehensiveness.csv')
    comprehensiveness_df.to_csv(comprehensiveness_path, index=False)
    print(f"  Saved: {comprehensiveness_path}")

    # --- Table 3: Summary Statistics ---
    aopc_results = compute_aopc(comprehensiveness_results)

    summary_df_data = []
    for method in METHODS:
        row = {
            'Method': method,
            'AOPC': f"{aopc_results[method]:.4f}",
            'Sparsity': f"{np.mean(sparsity_results[method]):.4f}",
            'Expl_Time_s': f"{np.mean(explanation_times[method]):.4f}",
            'Expl_Time_std': f"{np.std(explanation_times[method]):.4f}",
        }
        # Add mean sufficiency and comprehensiveness
        mean_suff = np.mean([np.mean(sufficiency_results[method][k]) for k in K_VALUES])
        mean_comp = np.mean([np.mean(comprehensiveness_results[method][k]) for k in K_VALUES[:-1]])
        row['Mean_Sufficiency'] = f"{mean_suff:.4f}"
        row['Mean_Comprehensiveness'] = f"{mean_comp:.4f}"
        summary_df_data.append(row)

    summary_df = pd.DataFrame(summary_df_data)
    summary_path = os.path.join(BASE_OUTPUT_DIR, 'results', 'exp1_table3_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # --- Save computational metrics ---
    total_time = time.time() - experiment_start_time

    computational_metrics = {
        'experiment_runtime_seconds': total_time,
        'device': str(DEVICE),
        'test_samples': len(test_items),
        'seq_len': seq_len,
        'input_dim': input_dim,
        'k_values': K_VALUES,
        'random_seed': RANDOM_SEED,
        'models': model_metrics,
        'explanation_times': {
            method: {
                'mean_seconds': float(np.mean(times)),
                'std_seconds': float(np.std(times)),
                'total_samples': len(times),
            }
            for method, times in explanation_times.items()
        },
    }

    metrics_path = os.path.join(BASE_OUTPUT_DIR, 'results', 'exp1_computational_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(computational_metrics, f, indent=2, default=str)
    print(f"  Saved: {metrics_path}")

    # ==========================================================================
    # STEP 6: GENERATE FIGURES
    # ==========================================================================
    print("\n[STEP 6/6] Generating figures...")

    # Color palette
    colors = {
        'TCDN': '#2ecc71',  # Green (our method)
        'TCDNFaithful': '#27ae60',  # Dark Green (our improved method)
        'TCDNFaithful+IG': '#1e8449',  # Darker Green (hybrid - best of both)
        'IntegratedGradients': '#3498db',  # Blue
        'GradientSHAP': '#9b59b6',  # Purple
        'Saliency': '#e74c3c',  # Red
        'SmoothGrad': '#f39c12',  # Orange
        'Attention': '#1abc9c',  # Teal
        'Random': '#95a5a6',  # Gray
    }

    # --- Figure 1: Sufficiency Curves ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in METHODS:
        means = [np.mean(sufficiency_results[method][k]) for k in K_VALUES]
        stds = [np.std(sufficiency_results[method][k]) for k in K_VALUES]

        linewidth = 3 if method in ['TCDN', 'TCDNFaithful', 'TCDNFaithful+IG'] else 1.5
        marker = 'o' if method == 'TCDN' else ('D' if method == 'TCDNFaithful' else ('^' if method == 'TCDNFaithful+IG' else 's'))

        ax.errorbar(K_VALUES, means, yerr=stds, label=method,
                    color=colors[method], linewidth=linewidth, marker=marker,
                    capsize=3, markersize=6)

    ax.set_xlabel('Top-k% Features Kept', fontsize=12)
    ax.set_ylabel('Prediction Probability (Sufficiency)', fontsize=12)
    ax.set_title('Sufficiency: Accuracy Using Only Top-k% Important Features', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)

    fig1_png = os.path.join(BASE_OUTPUT_DIR, 'figures', 'exp1_fig1_sufficiency_curves.png')
    fig1_pdf = os.path.join(BASE_OUTPUT_DIR, 'figures', 'exp1_fig1_sufficiency_curves.pdf')
    plt.savefig(fig1_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig1_pdf, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig1_png}")

    # --- Figure 2: Comprehensiveness Curves ---
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values_comp = K_VALUES[:-1]  # Exclude 100%

    for method in METHODS:
        means = [np.mean(comprehensiveness_results[method][k]) for k in k_values_comp]
        stds = [np.std(comprehensiveness_results[method][k]) for k in k_values_comp]

        linewidth = 3 if method in ['TCDN', 'TCDNFaithful', 'TCDNFaithful+IG'] else 1.5
        marker = 'o' if method == 'TCDN' else ('D' if method == 'TCDNFaithful' else ('^' if method == 'TCDNFaithful+IG' else 's'))

        ax.errorbar(k_values_comp, means, yerr=stds, label=method,
                    color=colors[method], linewidth=linewidth, marker=marker,
                    capsize=3, markersize=6)

    ax.set_xlabel('Top-k% Features Removed', fontsize=12)
    ax.set_ylabel('Probability Drop (Comprehensiveness)', fontsize=12)
    ax.set_title('Comprehensiveness: Prediction Drop When Removing Top-k% Features', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values_comp)

    fig2_png = os.path.join(BASE_OUTPUT_DIR, 'figures', 'exp1_fig2_comprehensiveness_curves.png')
    fig2_pdf = os.path.join(BASE_OUTPUT_DIR, 'figures', 'exp1_fig2_comprehensiveness_curves.pdf')
    plt.savefig(fig2_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig2_pdf, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig2_png}")

    # --- Figure 3: Radar Chart ---
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Metrics for radar chart
    metrics_names = ['AOPC', 'Suff@10%', 'Suff@50%', 'Comp@10%', 'Comp@50%', '1-Sparsity']
    num_metrics = len(metrics_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for method in METHODS:
        values = [
            aopc_results[method],
            np.mean(sufficiency_results[method][10]),
            np.mean(sufficiency_results[method][50]),
            np.mean(comprehensiveness_results[method][10]),
            np.mean(comprehensiveness_results[method][50]),
            1 - np.mean(sparsity_results[method]),  # 1-Sparsity so higher is better
        ]
        values += values[:1]  # Complete the circle

        linewidth = 3 if method in ['TCDN', 'TCDNFaithful', 'TCDNFaithful+IG'] else 1.5
        ax.plot(angles, values, label=method, color=colors[method], linewidth=linewidth)
        ax.fill(angles, values, color=colors[method], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_title('Faithfulness Metrics Comparison', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

    fig3_png = os.path.join(BASE_OUTPUT_DIR, 'figures', 'exp1_fig3_radar.png')
    fig3_pdf = os.path.join(BASE_OUTPUT_DIR, 'figures', 'exp1_fig3_radar.pdf')
    plt.savefig(fig3_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig3_pdf, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig3_png}")

    # ==========================================================================
    # PRINT SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    print("\n--- KEY RESULTS ---")
    print("\nModel Accuracies:")
    for model_name, metrics in model_metrics.items():
        if 'accuracy' in metrics:
            print(f"  {model_name}: {metrics['accuracy']:.4f}")

    print("\nFaithfulness Ranking (by AOPC, higher is better):")
    sorted_methods = sorted(aopc_results.items(), key=lambda x: x[1], reverse=True)
    for rank, (method, aopc) in enumerate(sorted_methods, 1):
        if method == 'TCDNFaithful+IG':
            marker = " <-- HYBRID (IG + Concepts)"
        elif method == 'TCDNFaithful':
            marker = " <-- OUR IMPROVED METHOD"
        elif method == 'TCDN':
            marker = " <-- OUR BASE METHOD"
        else:
            marker = ""
        print(f"  {rank}. {method}: {aopc:.4f}{marker}")

    print("\nMean Explanation Time (seconds):")
    for method in METHODS:
        mean_time = np.mean(explanation_times[method])
        print(f"  {method}: {mean_time:.4f}s")

    return {
        'sufficiency': sufficiency_results,
        'comprehensiveness': comprehensiveness_results,
        'aopc': aopc_results,
        'sparsity': sparsity_results,
        'explanation_times': explanation_times,
        'model_metrics': model_metrics,
    }


if __name__ == '__main__':
    results = run_experiment()
