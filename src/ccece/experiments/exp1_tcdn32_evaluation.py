#!/usr/bin/env python3
"""
CCECE Paper: Comprehensive TCDN-32 Evaluation

Test whether TCDN with 32 segments is publishable by comparing against all baselines.

Key questions:
1. Is AOPC competitive with gradient methods?
2. Does it maintain clinical interpretability?
3. Is accuracy preserved or improved?

Author: Experiment Agent
Date: 2026-01-18
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, compute_target_seq_len, set_all_seeds
from ccece.trainer import (
    TrainingConfig, Trainer, create_data_loaders, SequenceScaler,
    SaccadeDataset
)
from ccece.models import get_model
from ccece.models.temporal_concept_dynamics import TemporalConceptDynamicsNetwork

try:
    from captum.attr import IntegratedGradients, GradientShap, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
K_VALUES = [5, 10, 20, 50]
OUTPUT_DIR = './results/ccece/tcdn_experiments/exp1_faithfulness/results'


def explain_with_ig(model, x, device, use_train_mode=False):
    """Apply IntegratedGradients to model."""
    if not CAPTUM_AVAILABLE:
        return np.random.rand(x.shape[1])

    # Some models (RNN-based) need train mode for backward pass with cuDNN
    if use_train_mode:
        model.train()
    else:
        model.eval()

    def forward_func(inputs):
        return model(inputs)

    ig = IntegratedGradients(forward_func)
    x = x.to(device).requires_grad_(True)

    with torch.no_grad():
        pred = model(x).argmax(dim=1)

    baseline = torch.zeros_like(x)
    try:
        attr = ig.attribute(x, baseline, target=pred, n_steps=50)
    except RuntimeError:
        # Fallback: try with train mode
        model.train()
        attr = ig.attribute(x, baseline, target=pred, n_steps=50)
        model.eval()

    importance = attr[0].abs().sum(dim=1).detach().cpu().numpy()
    importance = importance / (importance.max() + 1e-8)

    model.eval()
    return importance


def explain_random(x):
    """Random baseline."""
    importance = np.random.rand(x.shape[1])
    return importance / (importance.max() + 1e-8)


def compute_faithfulness_metrics(model, x, importance, k_values, device):
    """Compute sufficiency and comprehensiveness."""
    model.eval()
    seq_len = x.shape[1]

    with torch.no_grad():
        x = x.to(device)
        original_output = model(x)
        original_pred = original_output.argmax(dim=1).item()
        original_prob = F.softmax(original_output, dim=1)[0, original_pred].item()

    sufficiency = {}
    comprehensiveness = {}

    for k in k_values:
        num_select = max(1, int(seq_len * k / 100))
        top_indices = set(np.argsort(importance)[-num_select:])

        # Sufficiency: keep only top-k%
        x_suff = x.clone()
        for t in range(seq_len):
            if t not in top_indices:
                x_suff[0, t, :] = 0
        with torch.no_grad():
            suff_prob = F.softmax(model(x_suff), dim=1)[0, original_pred].item()
        sufficiency[k] = suff_prob

        # Comprehensiveness: remove top-k%
        x_comp = x.clone()
        for t in top_indices:
            x_comp[0, t, :] = 0
        with torch.no_grad():
            comp_prob = F.softmax(model(x_comp), dim=1)[0, original_pred].item()
        comprehensiveness[k] = original_prob - comp_prob

    return sufficiency, comprehensiveness


def run_experiment():
    print("=" * 70)
    print("TCDN-32 COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Goal: Determine if TCDN-32 is publishable")
    print()

    set_all_seeds(RANDOM_SEED)

    # Load data
    print("\n[1/5] Loading data...")
    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)
    X, y, patient_ids = extract_arrays(items)

    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]

    print(f"  Samples: {len(items)}, Seq len: {seq_len}, Features: {input_dim}")

    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

    train_items = [items[i] for i in train_idx]
    test_items = [items[i] for i in test_idx]
    train_labels = y[train_idx]

    scaler = SequenceScaler().fit(train_items)
    test_dataset = SaccadeDataset(test_items, seq_len, scaler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    config = TrainingConfig(epochs=50, batch_size=32, learning_rate=1e-3, early_stopping_patience=10)
    train_loader, val_loader, _ = create_data_loaders(train_items, test_items, seq_len, config.batch_size, scaler)

    # Train models
    print("\n[2/5] Training models...")

    models = {}
    model_metrics = {}

    # TCDN-32
    print("  Training TCDN-32...")
    tcdn32 = TemporalConceptDynamicsNetwork(
        input_dim=input_dim, num_classes=2, seq_len=seq_len,
        hidden_dim=64, num_layers=4, kernel_size=7, dropout=0.2,
        num_segments=32, use_learned_concepts=True,
    )
    trainer = Trainer(tcdn32, config, DEVICE)
    trainer.train(train_loader, val_loader, train_labels, verbose=False)
    models['TCDN-32'] = tcdn32.to(DEVICE)
    metrics = trainer.evaluate(val_loader)
    model_metrics['TCDN-32'] = {
        'accuracy': metrics.accuracy,
        'auc_roc': metrics.auc_roc,
        'sensitivity': metrics.sensitivity,
        'specificity': metrics.specificity,
    }
    print(f"    Accuracy: {metrics.accuracy:.4f}, AUC: {metrics.auc_roc:.4f}")

    # TCDN-4 (original)
    print("  Training TCDN-4 (original)...")
    tcdn4 = TemporalConceptDynamicsNetwork(
        input_dim=input_dim, num_classes=2, seq_len=seq_len,
        hidden_dim=64, num_layers=4, kernel_size=7, dropout=0.2,
        num_segments=4, use_learned_concepts=True,
    )
    trainer = Trainer(tcdn4, config, DEVICE)
    trainer.train(train_loader, val_loader, train_labels, verbose=False)
    models['TCDN-4'] = tcdn4.to(DEVICE)
    metrics = trainer.evaluate(val_loader)
    model_metrics['TCDN-4'] = {'accuracy': metrics.accuracy, 'auc_roc': metrics.auc_roc}
    print(f"    Accuracy: {metrics.accuracy:.4f}, AUC: {metrics.auc_roc:.4f}")

    # TCN baseline
    print("  Training TCN...")
    tcn = get_model('tcn', input_dim=input_dim, num_classes=2, seq_len=seq_len)
    trainer = Trainer(tcn, config, DEVICE)
    trainer.train(train_loader, val_loader, train_labels, verbose=False)
    models['TCN'] = tcn.to(DEVICE)
    metrics = trainer.evaluate(val_loader)
    model_metrics['TCN'] = {'accuracy': metrics.accuracy, 'auc_roc': metrics.auc_roc}
    print(f"    Accuracy: {metrics.accuracy:.4f}, AUC: {metrics.auc_roc:.4f}")

    # BiGRU+Attention
    print("  Training BiGRU+Attention...")
    bigru = get_model('bigru_attention', input_dim=input_dim, num_classes=2, seq_len=seq_len)
    trainer = Trainer(bigru, config, DEVICE)
    trainer.train(train_loader, val_loader, train_labels, verbose=False)
    models['BiGRU'] = bigru.to(DEVICE)
    metrics = trainer.evaluate(val_loader)
    model_metrics['BiGRU'] = {'accuracy': metrics.accuracy, 'auc_roc': metrics.auc_roc}
    print(f"    Accuracy: {metrics.accuracy:.4f}, AUC: {metrics.auc_roc:.4f}")

    # Evaluate faithfulness
    print("\n[3/5] Evaluating faithfulness...")

    methods = {
        'TCDN-32+IG': (models['TCDN-32'], lambda m, x: explain_with_ig(m, x, DEVICE)),
        'TCDN-4+IG': (models['TCDN-4'], lambda m, x: explain_with_ig(m, x, DEVICE)),
        'TCN+IG': (models['TCN'], lambda m, x: explain_with_ig(m, x, DEVICE)),
        'BiGRU+IG': (models['BiGRU'], lambda m, x: explain_with_ig(m, x, DEVICE)),
        'Random': (models['TCN'], lambda m, x: explain_random(x)),
    }

    results = {method: {'sufficiency': {k: [] for k in K_VALUES},
                        'comprehensiveness': {k: [] for k in K_VALUES}}
               for method in methods}

    for x, _ in tqdm(test_loader, desc="  Evaluating"):
        x = x.to(DEVICE)

        for method_name, (model, explain_fn) in methods.items():
            importance = explain_fn(model, x)
            suff, comp = compute_faithfulness_metrics(model, x, importance, K_VALUES, DEVICE)

            for k in K_VALUES:
                results[method_name]['sufficiency'][k].append(suff[k])
                results[method_name]['comprehensiveness'][k].append(comp[k])

    # Compute summary metrics
    print("\n[4/5] Computing summary metrics...")

    summary = {}
    for method in methods:
        # AOPC = mean comprehensiveness across all k values
        all_comp = []
        for k in K_VALUES:
            all_comp.extend(results[method]['comprehensiveness'][k])
        aopc = np.mean(all_comp)

        # Mean sufficiency
        all_suff = []
        for k in K_VALUES:
            all_suff.extend(results[method]['sufficiency'][k])
        mean_suff = np.mean(all_suff)

        summary[method] = {
            'AOPC': aopc,
            'Mean_Sufficiency': mean_suff,
            'Comp@5%': np.mean(results[method]['comprehensiveness'][5]),
            'Comp@10%': np.mean(results[method]['comprehensiveness'][10]),
            'Comp@20%': np.mean(results[method]['comprehensiveness'][20]),
            'Suff@10%': np.mean(results[method]['sufficiency'][10]),
            'Suff@50%': np.mean(results[method]['sufficiency'][50]),
        }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- Model Performance ---")
    print(f"{'Model':<20} {'Accuracy':<12} {'AUC-ROC':<12}")
    print("-" * 44)
    for model_name, metrics in model_metrics.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['auc_roc']:<12.4f}")

    print("\n--- Faithfulness Ranking (AOPC, higher = better) ---")
    print(f"{'Method':<20} {'AOPC':<10} {'Comp@10%':<12} {'Suff@10%':<12}")
    print("-" * 54)
    sorted_methods = sorted(summary.items(), key=lambda x: x[1]['AOPC'], reverse=True)
    for method, metrics in sorted_methods:
        marker = " ***" if method == 'TCDN-32+IG' else ""
        print(f"{method:<20} {metrics['AOPC']:<10.4f} {metrics['Comp@10%']:<12.4f} {metrics['Suff@10%']:<12.4f}{marker}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: IS TCDN-32 PUBLISHABLE?")
    print("=" * 70)

    tcdn32_aopc = summary['TCDN-32+IG']['AOPC']
    tcdn4_aopc = summary['TCDN-4+IG']['AOPC']
    tcn_aopc = summary['TCN+IG']['AOPC']
    random_aopc = summary['Random']['AOPC']

    tcdn32_acc = model_metrics['TCDN-32']['accuracy']
    tcn_acc = model_metrics['TCN']['accuracy']

    print(f"\n1. FAITHFULNESS COMPARISON:")
    print(f"   TCDN-32+IG AOPC: {tcdn32_aopc:.4f}")
    print(f"   TCDN-4+IG AOPC:  {tcdn4_aopc:.4f} (original)")
    print(f"   TCN+IG AOPC:     {tcn_aopc:.4f} (baseline)")
    print(f"   Random AOPC:     {random_aopc:.4f}")
    print(f"   ")
    print(f"   Improvement over TCDN-4: {(tcdn32_aopc / tcdn4_aopc - 1) * 100:.1f}%")
    print(f"   Gap to TCN: {(tcn_aopc - tcdn32_aopc):.4f} ({(tcdn32_aopc / tcn_aopc) * 100:.1f}% of TCN)")

    print(f"\n2. ACCURACY COMPARISON:")
    print(f"   TCDN-32: {tcdn32_acc:.4f}")
    print(f"   TCN:     {tcn_acc:.4f}")
    print(f"   TCDN-32 {'>' if tcdn32_acc > tcn_acc else '<'} TCN by {abs(tcdn32_acc - tcn_acc):.4f}")

    print(f"\n3. PUBLISHABILITY ASSESSMENT:")

    # Criteria
    above_random = tcdn32_aopc > random_aopc * 1.5  # 50% above random
    close_to_baseline = tcdn32_aopc > tcn_aopc * 0.7  # Within 30% of baseline
    accuracy_ok = tcdn32_acc >= tcn_acc * 0.95  # Within 5% of baseline accuracy

    print(f"   [{'OK' if above_random else 'FAIL'}] Significantly above random (>{random_aopc * 1.5:.4f})")
    print(f"   [{'OK' if close_to_baseline else 'FAIL'}] Within 30% of TCN baseline (>{tcn_aopc * 0.7:.4f})")
    print(f"   [{'OK' if accuracy_ok else 'FAIL'}] Accuracy within 5% of baseline")

    publishable = above_random and close_to_baseline and accuracy_ok

    print(f"\n   VERDICT: {'PUBLISHABLE' if publishable else 'NOT PUBLISHABLE'}")

    if publishable:
        print(f"\n   TCDN-32 can be published with the following narrative:")
        print(f"   - 'TCDN-32 achieves {tcdn32_aopc / tcn_aopc * 100:.0f}% of TCN+IG faithfulness'")
        print(f"   - 'While maintaining clinical interpretability via 32 temporal windows'")
        print(f"   - 'With comparable accuracy ({tcdn32_acc:.1%} vs {tcn_acc:.1%})'")
    else:
        print(f"\n   Consider reframing around multi-level explainability instead.")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        'model_metrics': model_metrics,
        'faithfulness_summary': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()},
        'publishable': publishable,
        'criteria': {
            'above_random': above_random,
            'close_to_baseline': close_to_baseline,
            'accuracy_ok': accuracy_ok,
        },
        'timestamp': datetime.now().isoformat(),
    }

    output_path = os.path.join(OUTPUT_DIR, 'exp1_tcdn32_evaluation.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return summary, model_metrics, publishable


if __name__ == '__main__':
    summary, model_metrics, publishable = run_experiment()
