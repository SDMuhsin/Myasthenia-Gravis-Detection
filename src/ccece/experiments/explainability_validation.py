"""
CCECE Paper: Comprehensive Explainability Validation for TempProtoNet

This script addresses the explainability validation gaps:
1. Runs falsification tests across ALL folds (not just best)
2. Projects prototypes to nearest training exemplars
3. Generates visualizations showing what prototypes represent
4. Computes temporal interpretation metrics
5. Produces publication-ready explainability evidence

Output: Comprehensive validation report with visualizations
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import create_data_loaders
from ccece.models.temp_proto_net import TempProtoNet
from ccece.experiments.temp_proto_net_experiment import (
    TempProtoNetConfig, TempProtoNetTrainer,
    run_falsification_tests, store_training_embeddings
)

RANDOM_SEED = 42


@dataclass
class PrototypeProjection:
    """Result of projecting a prototype to nearest training samples."""
    prototype_idx: int
    prototype_class: int  # 0=HC, 1=MG
    prototype_class_name: str
    nearest_sample_indices: List[int]
    nearest_sample_distances: List[float]
    nearest_sample_labels: List[int]
    alignment_rate: float  # % of nearest neighbors with same class as prototype
    exemplar_time_series: np.ndarray  # The actual time series of nearest exemplar


@dataclass
class FoldExplainabilityResults:
    """Explainability results for a single fold."""
    fold: int
    accuracy: float

    # Falsification tests
    diversity_pass: bool
    alignment_pass: bool
    ablation_pass: bool
    all_tests_pass: bool

    # Detailed metrics
    mean_pairwise_distance: float
    min_pairwise_distance: float
    alignment_ratio: float
    alignment_rate: float  # % samples closer to own-class prototypes
    max_prototype_importance: float

    # Prototype projections
    prototype_projections: List[Dict]
    prototype_alignment_rate: float  # % prototypes whose neighbors match their class


def run_comprehensive_validation(output_dir: str, verbose: bool = True):
    """
    Run comprehensive explainability validation across all folds.
    """
    set_all_seeds(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print("\n" + "="*70)
        print("COMPREHENSIVE EXPLAINABILITY VALIDATION")
        print("="*70)

    # Load data
    if verbose:
        print("\nLoading data...")
    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)

    X, y, patient_ids = extract_arrays(items)
    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]

    if verbose:
        print(f"Data: {len(items)} samples, seq_len={seq_len}, input_dim={input_dim}")

    # Configuration (same as baseline)
    config = TempProtoNetConfig(
        latent_dim=64,
        n_prototypes_per_class=5,
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=15,
        cluster_loss_weight=0.5,
        separation_loss_weight=0.1,
        n_folds=5,
    )

    # Cross-validation
    cv = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results: List[FoldExplainabilityResults] = []
    all_prototype_projections = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, patient_ids)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{config.n_folds}")
            print('='*60)

        # Prepare data
        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        train_labels = np.array([item['label'] for item in train_items])

        train_loader, val_loader, scaler = create_data_loaders(
            train_items, val_items, seq_len, config.batch_size
        )

        # Create and train model
        model = TempProtoNet(
            input_dim=input_dim,
            num_classes=2,
            seq_len=seq_len,
            latent_dim=config.latent_dim,
            n_prototypes_per_class=config.n_prototypes_per_class,
            encoder_hidden=config.encoder_hidden,
            encoder_layers=config.encoder_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )

        if verbose:
            print(f"Training model ({model.count_parameters():,} parameters)...")

        trainer = TempProtoNetTrainer(model, config, device)
        trainer.train(train_loader, val_loader, train_labels, verbose=verbose)

        # Evaluate accuracy
        metrics = trainer.evaluate(val_loader)

        # Store training embeddings and raw data for projection
        store_training_embeddings(model, train_loader, device)

        # Also store raw training data for visualization
        train_data_list = []
        train_label_list = []
        for batch_x, batch_y in train_loader:
            train_data_list.append(batch_x.numpy())
            train_label_list.append(batch_y.numpy())
        train_data = np.concatenate(train_data_list, axis=0)
        train_labels_arr = np.concatenate(train_label_list, axis=0)

        # Run falsification tests
        falsification = run_falsification_tests(model, val_loader, device)

        # Compute detailed alignment rate
        alignment_rate = compute_alignment_rate(model, val_loader, device)

        # Project prototypes to nearest training samples
        projections = project_prototypes_with_data(
            model, train_data, train_labels_arr, device
        )

        # Compute prototype alignment rate
        proto_alignment_rates = []
        for proj in projections:
            # What % of nearest neighbors have same class as prototype?
            same_class = sum(1 for lbl in proj['nearest_labels'] if lbl == proj['prototype_class'])
            rate = same_class / len(proj['nearest_labels'])
            proj['alignment_rate'] = rate
            proto_alignment_rates.append(rate)

        mean_proto_alignment = np.mean(proto_alignment_rates)

        if verbose:
            print(f"\n  Results:")
            print(f"    Accuracy: {metrics.accuracy:.4f}")
            print(f"    Diversity: {'PASS' if falsification.diversity_pass else 'FAIL'} "
                  f"(mean_dist={falsification.mean_pairwise_distance:.3f})")
            print(f"    Alignment: {'PASS' if falsification.alignment_pass else 'FAIL'} "
                  f"(rate={alignment_rate:.1%})")
            print(f"    Ablation: {'PASS' if falsification.ablation_pass else 'FAIL'} "
                  f"(max_imp={max(falsification.prototype_importance.values()):.1%})")
            print(f"    Proto Alignment: {mean_proto_alignment:.1%} of prototype neighbors match class")
            print(f"    ALL TESTS: {'PASS' if falsification.all_pass else 'FAIL'}")

        fold_result = FoldExplainabilityResults(
            fold=fold + 1,
            accuracy=metrics.accuracy,
            diversity_pass=falsification.diversity_pass,
            alignment_pass=falsification.alignment_pass,
            ablation_pass=falsification.ablation_pass,
            all_tests_pass=falsification.all_pass,
            mean_pairwise_distance=falsification.mean_pairwise_distance,
            min_pairwise_distance=falsification.min_pairwise_distance,
            alignment_ratio=falsification.alignment_ratio,
            alignment_rate=alignment_rate,
            max_prototype_importance=max(falsification.prototype_importance.values()),
            prototype_projections=projections,
            prototype_alignment_rate=mean_proto_alignment,
        )
        fold_results.append(fold_result)
        all_prototype_projections.append(projections)

        # Generate visualizations for this fold
        visualize_fold_prototypes(
            model, projections, train_data, fold + 1, output_dir
        )

    # Aggregate results
    summary = aggregate_results(fold_results, output_dir, verbose)

    # Generate summary visualization
    visualize_summary(fold_results, output_dir)

    return summary


def compute_alignment_rate(
    model: TempProtoNet,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """Compute % of samples closer to own-class prototypes."""
    model.eval()
    prototype_classes = model.get_prototype_classes()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            similarities = model.get_prototype_similarities(inputs)

            for i in range(inputs.size(0)):
                label = labels[i].item()
                sims = similarities[i]

                mg_proto_mask = (prototype_classes == 1)
                hc_proto_mask = (prototype_classes == 0)

                mg_sim = sims[mg_proto_mask].mean().item()
                hc_sim = sims[hc_proto_mask].mean().item()

                # Is sample closer to its own class?
                if label == 1 and mg_sim > hc_sim:
                    correct += 1
                elif label == 0 and hc_sim > mg_sim:
                    correct += 1
                total += 1

    return correct / total


def project_prototypes_with_data(
    model: TempProtoNet,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    device: torch.device,
    k: int = 5,
) -> List[Dict]:
    """Project prototypes and include actual time series data."""
    model.eval()

    # Move training embeddings to same device as model for projection
    if model._training_embeddings is not None:
        model._training_embeddings = model._training_embeddings.to(device)
        model._training_labels = model._training_labels.to(device)

    # Get projections using model's method
    projections = model.project_prototypes(k=k)

    # Enrich with actual data
    for proj in projections:
        proto_class = proj['prototype_class']
        nearest_idx = proj['nearest_indices'][0]  # Closest sample

        proj['prototype_class_name'] = 'MG' if proto_class == 1 else 'HC'
        proj['exemplar_data'] = train_data[nearest_idx]  # (seq_len, input_dim)
        proj['nearest_labels'] = train_labels[proj['nearest_indices']]

    return projections


def visualize_fold_prototypes(
    model: TempProtoNet,
    projections: List[Dict],
    train_data: np.ndarray,
    fold: int,
    output_dir: str,
):
    """Generate prototype visualization for a fold."""
    n_prototypes = len(projections)
    n_cols = 5
    n_rows = (n_prototypes + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    # Feature names for eye-tracking data
    feature_names = [
        'X position', 'Y position', 'Pupil', 'Velocity',
        'Acceleration', 'Fixation', 'Saccade', 'Smooth pursuit',
        'Blink', 'Head X', 'Head Y', 'Head Z',
        'Gaze X', 'Gaze Y'
    ]

    for idx, proj in enumerate(projections):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Get exemplar data
        exemplar = proj['exemplar_data']  # (seq_len, input_dim)
        proto_class = proj['prototype_class_name']
        alignment = proj.get('alignment_rate', 0)

        # Plot selected features (X position, Y position, Velocity)
        time = np.arange(exemplar.shape[0])
        ax.plot(time, exemplar[:, 0], label='X pos', alpha=0.7, linewidth=0.5)
        ax.plot(time, exemplar[:, 1], label='Y pos', alpha=0.7, linewidth=0.5)
        ax.plot(time, exemplar[:, 3], label='Velocity', alpha=0.7, linewidth=0.5)

        ax.set_title(f'Prototype {idx} ({proto_class})\nAlign: {alignment:.0%}', fontsize=10)
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Normalized value', fontsize=8)
        ax.tick_params(labelsize=7)

        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    plt.suptitle(f'Fold {fold}: Prototype Exemplars (Nearest Training Samples)', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'fold{fold}_prototype_exemplars.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_summary(fold_results: List[FoldExplainabilityResults], output_dir: str):
    """Generate summary visualization across all folds."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Accuracy and test pass rates per fold
    ax1 = axes[0, 0]
    folds = [r.fold for r in fold_results]
    accuracies = [r.accuracy for r in fold_results]
    all_pass = [1 if r.all_tests_pass else 0 for r in fold_results]

    x = np.arange(len(folds))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax1.bar(x + width/2, all_pass, width, label='All Tests Pass', color='green', alpha=0.7)

    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Value')
    ax1.set_title('Accuracy and Falsification Tests per Fold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds)
    ax1.legend()
    ax1.axhline(y=0.717, color='red', linestyle='--', label='Baseline (71.7%)')
    ax1.set_ylim(0, 1.1)

    # 2. Individual test pass rates
    ax2 = axes[0, 1]
    test_names = ['Diversity', 'Alignment', 'Ablation', 'All Pass']
    diversity_pass = sum(1 for r in fold_results if r.diversity_pass) / len(fold_results)
    alignment_pass = sum(1 for r in fold_results if r.alignment_pass) / len(fold_results)
    ablation_pass = sum(1 for r in fold_results if r.ablation_pass) / len(fold_results)
    all_pass_rate = sum(1 for r in fold_results if r.all_tests_pass) / len(fold_results)

    pass_rates = [diversity_pass, alignment_pass, ablation_pass, all_pass_rate]
    colors = ['green' if r >= 0.8 else 'orange' if r >= 0.6 else 'red' for r in pass_rates]

    bars = ax2.bar(test_names, pass_rates, color=colors)
    ax2.set_ylabel('Pass Rate Across Folds')
    ax2.set_title('Falsification Test Pass Rates')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')

    for bar, rate in zip(bars, pass_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.0%}', ha='center', fontsize=10)

    # 3. Prototype alignment rates
    ax3 = axes[1, 0]
    proto_align_rates = [r.prototype_alignment_rate for r in fold_results]
    sample_align_rates = [r.alignment_rate for r in fold_results]

    x = np.arange(len(folds))
    bars1 = ax3.bar(x - width/2, proto_align_rates, width, label='Prototype Alignment', color='purple')
    bars2 = ax3.bar(x + width/2, sample_align_rates, width, label='Sample Alignment', color='teal')

    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Alignment Rate')
    ax3.set_title('Class Alignment Rates')
    ax3.set_xticks(x)
    ax3.set_xticklabels(folds)
    ax3.legend()
    ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='70% threshold')
    ax3.set_ylim(0, 1.1)

    # 4. Prototype diversity metrics
    ax4 = axes[1, 1]
    mean_dists = [r.mean_pairwise_distance for r in fold_results]
    min_dists = [r.min_pairwise_distance for r in fold_results]

    x = np.arange(len(folds))
    bars1 = ax4.bar(x - width/2, mean_dists, width, label='Mean Distance', color='coral')
    bars2 = ax4.bar(x + width/2, min_dists, width, label='Min Distance', color='gold')

    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Distance')
    ax4.set_title('Prototype Diversity')
    ax4.set_xticks(x)
    ax4.set_xticklabels(folds)
    ax4.legend()
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Mean threshold')
    ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Min threshold')

    plt.suptitle('TempProtoNet Explainability Validation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'explainability_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def aggregate_results(
    fold_results: List[FoldExplainabilityResults],
    output_dir: str,
    verbose: bool
) -> Dict:
    """Aggregate results across folds and generate report."""

    # Compute aggregates
    n_folds = len(fold_results)

    mean_accuracy = np.mean([r.accuracy for r in fold_results])
    std_accuracy = np.std([r.accuracy for r in fold_results])

    diversity_pass_rate = sum(1 for r in fold_results if r.diversity_pass) / n_folds
    alignment_pass_rate = sum(1 for r in fold_results if r.alignment_pass) / n_folds
    ablation_pass_rate = sum(1 for r in fold_results if r.ablation_pass) / n_folds
    all_pass_rate = sum(1 for r in fold_results if r.all_tests_pass) / n_folds

    mean_alignment_rate = np.mean([r.alignment_rate for r in fold_results])
    mean_proto_alignment = np.mean([r.prototype_alignment_rate for r in fold_results])
    mean_diversity = np.mean([r.mean_pairwise_distance for r in fold_results])
    mean_max_importance = np.mean([r.max_prototype_importance for r in fold_results])

    # Determine overall verdict
    explainability_valid = (
        all_pass_rate >= 0.6 and  # At least 60% of folds pass all tests
        mean_alignment_rate >= 0.65 and  # Samples generally align with prototypes
        mean_proto_alignment >= 0.6  # Prototype neighbors match their class
    )

    summary = {
        'performance': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'fold_accuracies': [r.accuracy for r in fold_results],
        },
        'falsification_tests': {
            'diversity_pass_rate': diversity_pass_rate,
            'alignment_pass_rate': alignment_pass_rate,
            'ablation_pass_rate': ablation_pass_rate,
            'all_tests_pass_rate': all_pass_rate,
            'per_fold': [r.all_tests_pass for r in fold_results],
        },
        'explainability_metrics': {
            'mean_sample_alignment_rate': mean_alignment_rate,
            'mean_prototype_alignment_rate': mean_proto_alignment,
            'mean_prototype_diversity': mean_diversity,
            'mean_max_prototype_importance': mean_max_importance,
        },
        'verdict': {
            'explainability_valid': explainability_valid,
            'reasoning': generate_verdict_reasoning(
                all_pass_rate, mean_alignment_rate, mean_proto_alignment
            ),
        },
        'per_fold_results': [
            {
                'fold': r.fold,
                'accuracy': r.accuracy,
                'all_tests_pass': r.all_tests_pass,
                'alignment_rate': r.alignment_rate,
                'prototype_alignment_rate': r.prototype_alignment_rate,
            }
            for r in fold_results
        ],
        'timestamp': datetime.now().isoformat(),
    }

    if verbose:
        print("\n" + "="*70)
        print("EXPLAINABILITY VALIDATION SUMMARY")
        print("="*70)

        print(f"\n1. PERFORMANCE")
        print(f"   Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

        print(f"\n2. FALSIFICATION TESTS (Pass Rate Across {n_folds} Folds)")
        print(f"   Diversity:  {diversity_pass_rate:.0%} {'✓' if diversity_pass_rate >= 0.8 else '✗'}")
        print(f"   Alignment:  {alignment_pass_rate:.0%} {'✓' if alignment_pass_rate >= 0.8 else '✗'}")
        print(f"   Ablation:   {ablation_pass_rate:.0%} {'✓' if ablation_pass_rate >= 0.8 else '✗'}")
        print(f"   ALL PASS:   {all_pass_rate:.0%} {'✓' if all_pass_rate >= 0.6 else '✗'}")

        print(f"\n3. EXPLAINABILITY METRICS")
        print(f"   Sample Alignment Rate:    {mean_alignment_rate:.1%} (>70% = good)")
        print(f"   Prototype Alignment Rate: {mean_proto_alignment:.1%} (>60% = good)")
        print(f"   Prototype Diversity:      {mean_diversity:.3f} (>0.5 = good)")
        print(f"   Max Prototype Importance: {mean_max_importance:.1%} (>2% = good)")

        print(f"\n4. VERDICT")
        print(f"   Explainability Valid: {'YES' if explainability_valid else 'NO'}")
        print(f"   {summary['verdict']['reasoning']}")

        print("\n" + "="*70)

    # Save results
    results_path = os.path.join(output_dir, 'explainability_validation_results.json')

    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)

    if verbose:
        print(f"\nResults saved to: {results_path}")
        print(f"Visualizations saved to: {output_dir}/")

    return summary


def generate_verdict_reasoning(
    all_pass_rate: float,
    alignment_rate: float,
    proto_alignment: float,
) -> str:
    """Generate human-readable reasoning for verdict."""
    issues = []
    strengths = []

    if all_pass_rate >= 0.8:
        strengths.append(f"Falsification tests pass in {all_pass_rate:.0%} of folds")
    elif all_pass_rate >= 0.6:
        issues.append(f"Falsification tests pass in only {all_pass_rate:.0%} of folds (acceptable but not ideal)")
    else:
        issues.append(f"Falsification tests pass in only {all_pass_rate:.0%} of folds (needs improvement)")

    if alignment_rate >= 0.7:
        strengths.append(f"Samples align with own-class prototypes {alignment_rate:.0%} of the time")
    else:
        issues.append(f"Sample-prototype alignment is weak ({alignment_rate:.0%})")

    if proto_alignment >= 0.7:
        strengths.append(f"Prototype neighbors match their class {proto_alignment:.0%} of the time")
    elif proto_alignment >= 0.6:
        strengths.append(f"Prototype neighbors mostly match their class ({proto_alignment:.0%})")
    else:
        issues.append(f"Prototype neighbors often don't match their class ({proto_alignment:.0%})")

    if not issues:
        return "Strong explainability: " + "; ".join(strengths)
    elif not strengths:
        return "Weak explainability: " + "; ".join(issues)
    else:
        return f"Mixed: Strengths - {'; '.join(strengths)}. Issues - {'; '.join(issues)}"


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/ccece/explainability_validation/{timestamp}"

    summary = run_comprehensive_validation(output_dir, verbose=True)
