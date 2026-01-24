"""
Prototype Alignment Investigation: Phase 1 - Observation

This script investigates why TempProtoNet prototype alignment is only 49.6%
while the model achieves 71% accuracy.

Goal: Observe what exists before hypothesizing.
- Per-prototype alignment rates
- Distribution of neighbor labels for each prototype
- Patterns across folds

Output: Data for investigation report, no speculation.
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import create_data_loaders
from ccece.models.temp_proto_net import TempProtoNet
from ccece.experiments.temp_proto_net_experiment import (
    TempProtoNetConfig, TempProtoNetTrainer, store_training_embeddings
)

RANDOM_SEED = 42
K_NEIGHBORS = 10  # Number of nearest neighbors to analyze


def compute_detailed_prototype_alignment(
    model: TempProtoNet,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    device: torch.device,
    k: int = K_NEIGHBORS,
) -> List[Dict]:
    """
    Compute detailed alignment info for each prototype.

    Returns:
        List of dicts per prototype with:
        - prototype_idx
        - prototype_class
        - neighbor_labels (list of k labels)
        - neighbor_distances (list of k distances)
        - alignment_rate (% matching prototype class)
        - class_distribution (counts per class)
    """
    model.eval()

    # Get stored training embeddings
    if model._training_embeddings is None:
        raise ValueError("Training embeddings not stored")

    embeddings = model._training_embeddings.to(device)
    prototypes = model.prototype_layer.prototypes.detach().to(device)
    prototype_classes = model.prototype_layer.prototype_class

    results = []

    for proto_idx in range(prototypes.size(0)):
        proto = prototypes[proto_idx:proto_idx+1]  # (1, latent_dim)
        proto_class = prototype_classes[proto_idx].item()

        # Compute distances to all training samples
        dists = torch.cdist(proto, embeddings).squeeze(0)  # (n_samples,)

        # Get k nearest
        topk_dists, topk_indices = torch.topk(dists, k, largest=False)

        neighbor_labels = train_labels[topk_indices.cpu().numpy()]
        neighbor_distances = topk_dists.cpu().numpy()

        # Count class distribution
        n_hc = sum(1 for lbl in neighbor_labels if lbl == 0)
        n_mg = sum(1 for lbl in neighbor_labels if lbl == 1)

        # Compute alignment rate
        n_same_class = sum(1 for lbl in neighbor_labels if lbl == proto_class)
        alignment_rate = n_same_class / k

        results.append({
            'prototype_idx': proto_idx,
            'prototype_class': proto_class,
            'prototype_class_name': 'MG' if proto_class == 1 else 'HC',
            'neighbor_labels': neighbor_labels.tolist(),
            'neighbor_distances': neighbor_distances.tolist(),
            'alignment_rate': alignment_rate,
            'n_hc_neighbors': n_hc,
            'n_mg_neighbors': n_mg,
        })

    return results


def analyze_prototype_positions(
    model: TempProtoNet,
    device: torch.device,
) -> Dict:
    """
    Analyze the geometric position of prototypes in latent space.

    Returns:
        Dict with prototype position metrics.
    """
    model.eval()

    prototypes = model.prototype_layer.prototypes.detach().to(device)
    prototype_classes = model.prototype_layer.prototype_class

    # Compute pairwise distances between prototypes
    proto_dists = torch.cdist(prototypes, prototypes).cpu().numpy()

    # Separate into intra-class and inter-class
    n_protos = prototypes.size(0)
    intra_class_dists = []
    inter_class_dists = []

    for i in range(n_protos):
        for j in range(i+1, n_protos):
            dist = proto_dists[i, j]
            if prototype_classes[i] == prototype_classes[j]:
                intra_class_dists.append(dist)
            else:
                inter_class_dists.append(dist)

    # Also compute distance of each prototype to class centroids
    if model._training_embeddings is not None:
        embeddings = model._training_embeddings.to(device)
        labels = model._training_labels.to(device)

        hc_mask = (labels == 0)
        mg_mask = (labels == 1)

        hc_centroid = embeddings[hc_mask].mean(dim=0, keepdim=True) if hc_mask.any() else None
        mg_centroid = embeddings[mg_mask].mean(dim=0, keepdim=True) if mg_mask.any() else None

        proto_to_hc_centroid = []
        proto_to_mg_centroid = []

        for proto_idx in range(n_protos):
            proto = prototypes[proto_idx:proto_idx+1]
            if hc_centroid is not None:
                proto_to_hc_centroid.append(torch.cdist(proto, hc_centroid).item())
            if mg_centroid is not None:
                proto_to_mg_centroid.append(torch.cdist(proto, mg_centroid).item())

    return {
        'mean_intra_class_dist': float(np.mean(intra_class_dists)),
        'mean_inter_class_dist': float(np.mean(inter_class_dists)),
        'min_inter_class_dist': float(np.min(inter_class_dists)),
        'proto_to_hc_centroid': proto_to_hc_centroid,
        'proto_to_mg_centroid': proto_to_mg_centroid,
        'prototype_classes': prototype_classes.cpu().numpy().tolist(),
    }


def analyze_embedding_distribution(
    model: TempProtoNet,
    device: torch.device,
) -> Dict:
    """
    Analyze how training embeddings are distributed relative to prototypes.
    """
    model.eval()

    if model._training_embeddings is None:
        raise ValueError("Training embeddings not stored")

    embeddings = model._training_embeddings.to(device)
    labels = model._training_labels.to(device)
    prototypes = model.prototype_layer.prototypes.detach().to(device)
    prototype_classes = model.prototype_layer.prototype_class

    # For each sample, find its nearest prototype
    dists_to_protos = torch.cdist(embeddings, prototypes)  # (n_samples, n_protos)
    nearest_proto = torch.argmin(dists_to_protos, dim=1)  # (n_samples,)

    # Count how many samples are assigned to each prototype
    proto_counts = {}
    proto_correct_counts = {}

    for proto_idx in range(prototypes.size(0)):
        mask = (nearest_proto == proto_idx)
        count = mask.sum().item()
        proto_counts[proto_idx] = count

        # Of those, how many have matching class?
        if count > 0:
            proto_class = prototype_classes[proto_idx].item()
            sample_labels = labels[mask]
            correct = (sample_labels == proto_class).sum().item()
            proto_correct_counts[proto_idx] = correct
        else:
            proto_correct_counts[proto_idx] = 0

    # Compute class-wise stats
    hc_embeddings = embeddings[labels == 0]
    mg_embeddings = embeddings[labels == 1]

    # Distance from each class to same-class vs other-class prototypes
    hc_protos = prototypes[prototype_classes == 0]
    mg_protos = prototypes[prototype_classes == 1]

    hc_to_hc_protos = torch.cdist(hc_embeddings, hc_protos).mean().item() if len(hc_embeddings) > 0 else 0
    hc_to_mg_protos = torch.cdist(hc_embeddings, mg_protos).mean().item() if len(hc_embeddings) > 0 else 0
    mg_to_mg_protos = torch.cdist(mg_embeddings, mg_protos).mean().item() if len(mg_embeddings) > 0 else 0
    mg_to_hc_protos = torch.cdist(mg_embeddings, hc_protos).mean().item() if len(mg_embeddings) > 0 else 0

    return {
        'proto_sample_counts': proto_counts,
        'proto_correct_counts': proto_correct_counts,
        'hc_to_hc_protos_dist': hc_to_hc_protos,
        'hc_to_mg_protos_dist': hc_to_mg_protos,
        'mg_to_mg_protos_dist': mg_to_mg_protos,
        'mg_to_hc_protos_dist': mg_to_hc_protos,
    }


def run_investigation(output_dir: str, verbose: bool = True):
    """
    Run Phase 1 investigation: observe per-prototype alignment.
    """
    set_all_seeds(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print("\n" + "="*70)
        print("PROTOTYPE ALIGNMENT INVESTIGATION - PHASE 1: OBSERVATION")
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
        print(f"Class distribution: HC={sum(y==0)}, MG={sum(y==1)}")

    # Configuration
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

    cv = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=RANDOM_SEED)

    all_fold_results = []

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

        # Extract train data as arrays
        train_data_list = []
        train_label_list = []
        for batch_x, batch_y in train_loader:
            train_data_list.append(batch_x.numpy())
            train_label_list.append(batch_y.numpy())
        train_data = np.concatenate(train_data_list, axis=0)
        train_labels_arr = np.concatenate(train_label_list, axis=0)

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

        # Store training embeddings
        store_training_embeddings(model, train_loader, device)

        # Evaluate accuracy
        metrics = trainer.evaluate(val_loader)

        # DETAILED INVESTIGATION
        if verbose:
            print(f"\n  Running detailed prototype analysis...")

        # 1. Per-prototype alignment
        proto_alignment = compute_detailed_prototype_alignment(
            model, train_data, train_labels_arr, device, k=K_NEIGHBORS
        )

        # 2. Prototype positions
        proto_positions = analyze_prototype_positions(model, device)

        # 3. Embedding distribution
        embedding_dist = analyze_embedding_distribution(model, device)

        fold_result = {
            'fold': fold + 1,
            'accuracy': metrics.accuracy,
            'per_prototype_alignment': proto_alignment,
            'prototype_positions': proto_positions,
            'embedding_distribution': embedding_dist,
        }
        all_fold_results.append(fold_result)

        # Print per-prototype alignment
        if verbose:
            print(f"\n  Per-Prototype Alignment (k={K_NEIGHBORS}):")
            print(f"  {'Proto':<6} {'Class':<6} {'Align%':<8} {'HC':<4} {'MG':<4} {'Distances':<30}")
            print(f"  {'-'*60}")

            for p in proto_alignment:
                dists_str = ', '.join([f'{d:.3f}' for d in p['neighbor_distances'][:5]])
                print(f"  {p['prototype_idx']:<6} {p['prototype_class_name']:<6} "
                      f"{p['alignment_rate']*100:<8.1f} {p['n_hc_neighbors']:<4} "
                      f"{p['n_mg_neighbors']:<4} [{dists_str}, ...]")

            mean_align = np.mean([p['alignment_rate'] for p in proto_alignment])
            print(f"\n  Mean prototype alignment: {mean_align:.1%}")
            print(f"  Accuracy: {metrics.accuracy:.1%}")

        # Print embedding distribution info
        if verbose:
            print(f"\n  Embedding-Prototype Distances:")
            print(f"    HC samples to HC prototypes: {embedding_dist['hc_to_hc_protos_dist']:.4f}")
            print(f"    HC samples to MG prototypes: {embedding_dist['hc_to_mg_protos_dist']:.4f}")
            print(f"    MG samples to MG prototypes: {embedding_dist['mg_to_mg_protos_dist']:.4f}")
            print(f"    MG samples to HC prototypes: {embedding_dist['mg_to_hc_protos_dist']:.4f}")

            print(f"\n  Samples assigned to each prototype (nearest):")
            n_protos = len(proto_positions['prototype_classes'])
            for proto_idx in range(n_protos):
                count = embedding_dist['proto_sample_counts'].get(proto_idx, 0)
                correct = embedding_dist['proto_correct_counts'].get(proto_idx, 0)
                pct = correct / count * 100 if count > 0 else 0
                proto_class = 'HC' if proto_positions['prototype_classes'][proto_idx] == 0 else 'MG'
                print(f"    Proto {proto_idx} ({proto_class}): {count} samples, {correct} correct ({pct:.1f}%)")

    # Aggregate and save results
    summary = aggregate_investigation_results(all_fold_results, output_dir, verbose)

    # Generate visualizations
    generate_investigation_visualizations(all_fold_results, output_dir)

    return summary


def aggregate_investigation_results(
    fold_results: List[Dict],
    output_dir: str,
    verbose: bool
) -> Dict:
    """
    Aggregate investigation results across folds.
    """
    # Aggregate per-prototype alignment across folds
    n_prototypes = 10
    n_folds = len(fold_results)

    proto_alignment_matrix = np.zeros((n_folds, n_prototypes))
    proto_hc_count_matrix = np.zeros((n_folds, n_prototypes))
    proto_mg_count_matrix = np.zeros((n_folds, n_prototypes))

    for fold_idx, fr in enumerate(fold_results):
        for p in fr['per_prototype_alignment']:
            proto_idx = p['prototype_idx']
            proto_alignment_matrix[fold_idx, proto_idx] = p['alignment_rate']
            proto_hc_count_matrix[fold_idx, proto_idx] = p['n_hc_neighbors']
            proto_mg_count_matrix[fold_idx, proto_idx] = p['n_mg_neighbors']

    # Mean alignment per prototype across folds
    mean_alignment_per_proto = proto_alignment_matrix.mean(axis=0)
    std_alignment_per_proto = proto_alignment_matrix.std(axis=0)

    # Identify good vs poor prototypes
    good_protos = np.where(mean_alignment_per_proto >= 0.7)[0].tolist()
    marginal_protos = np.where((mean_alignment_per_proto >= 0.5) & (mean_alignment_per_proto < 0.7))[0].tolist()
    poor_protos = np.where(mean_alignment_per_proto < 0.5)[0].tolist()

    # Get prototype class assignments (same across folds)
    proto_classes = fold_results[0]['prototype_positions']['prototype_classes']

    summary = {
        'meta': {
            'k_neighbors': K_NEIGHBORS,
            'n_folds': n_folds,
            'timestamp': datetime.now().isoformat(),
        },
        'per_prototype_summary': {
            'mean_alignment': mean_alignment_per_proto.tolist(),
            'std_alignment': std_alignment_per_proto.tolist(),
            'prototype_classes': proto_classes,
            'good_prototypes': good_protos,
            'marginal_prototypes': marginal_protos,
            'poor_prototypes': poor_protos,
        },
        'per_fold_results': fold_results,
        'alignment_matrix': proto_alignment_matrix.tolist(),
        'overall': {
            'mean_accuracy': np.mean([fr['accuracy'] for fr in fold_results]),
            'mean_prototype_alignment': proto_alignment_matrix.mean(),
            'n_good_protos': len(good_protos),
            'n_marginal_protos': len(marginal_protos),
            'n_poor_protos': len(poor_protos),
        }
    }

    if verbose:
        print("\n" + "="*70)
        print("INVESTIGATION SUMMARY")
        print("="*70)

        print(f"\n1. OVERALL METRICS")
        print(f"   Mean accuracy: {summary['overall']['mean_accuracy']:.1%}")
        print(f"   Mean prototype alignment: {summary['overall']['mean_prototype_alignment']:.1%}")

        print(f"\n2. PER-PROTOTYPE ALIGNMENT (across {n_folds} folds, k={K_NEIGHBORS})")
        print(f"   {'Proto':<6} {'Class':<6} {'Mean':<8} {'Std':<8} {'Status':<10}")
        print(f"   {'-'*40}")

        for i in range(n_prototypes):
            proto_class = 'HC' if proto_classes[i] == 0 else 'MG'
            mean_align = mean_alignment_per_proto[i]
            std_align = std_alignment_per_proto[i]

            if i in good_protos:
                status = 'GOOD'
            elif i in marginal_protos:
                status = 'MARGINAL'
            else:
                status = 'POOR'

            print(f"   {i:<6} {proto_class:<6} {mean_align*100:<8.1f} {std_align*100:<8.1f} {status:<10}")

        print(f"\n3. PROTOTYPE CATEGORIES")
        print(f"   Good (>=70%): {len(good_protos)} prototypes: {good_protos}")
        print(f"   Marginal (50-70%): {len(marginal_protos)} prototypes: {marginal_protos}")
        print(f"   Poor (<50%): {len(poor_protos)} prototypes: {poor_protos}")

        # Check if there's a pattern by class
        hc_protos = [i for i in range(n_prototypes) if proto_classes[i] == 0]
        mg_protos = [i for i in range(n_prototypes) if proto_classes[i] == 1]

        hc_mean_align = np.mean([mean_alignment_per_proto[i] for i in hc_protos])
        mg_mean_align = np.mean([mean_alignment_per_proto[i] for i in mg_protos])

        print(f"\n4. ALIGNMENT BY PROTOTYPE CLASS")
        print(f"   HC prototypes (0-4): mean alignment = {hc_mean_align:.1%}")
        print(f"   MG prototypes (5-9): mean alignment = {mg_mean_align:.1%}")

    # Save results
    results_path = os.path.join(output_dir, 'investigation_results.json')

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

    return summary


def generate_investigation_visualizations(
    fold_results: List[Dict],
    output_dir: str,
):
    """
    Generate visualizations for Phase 1 investigation.
    """
    n_prototypes = 10
    n_folds = len(fold_results)

    # Extract alignment matrix
    proto_alignment_matrix = np.zeros((n_folds, n_prototypes))
    for fold_idx, fr in enumerate(fold_results):
        for p in fr['per_prototype_alignment']:
            proto_idx = p['prototype_idx']
            proto_alignment_matrix[fold_idx, proto_idx] = p['alignment_rate']

    # Get prototype classes
    proto_classes = fold_results[0]['prototype_positions']['prototype_classes']

    # Figure 1: Heatmap of per-prototype alignment across folds
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1.1 Heatmap
    ax1 = axes[0, 0]
    im = ax1.imshow(proto_alignment_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax1.set_xlabel('Prototype Index')
    ax1.set_ylabel('Fold')
    ax1.set_title('Prototype Alignment Rate per Fold')
    ax1.set_xticks(range(n_prototypes))
    ax1.set_xticklabels([f'P{i}\n({"HC" if proto_classes[i]==0 else "MG"})' for i in range(n_prototypes)])
    ax1.set_yticks(range(n_folds))
    ax1.set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])

    # Add text annotations
    for i in range(n_folds):
        for j in range(n_prototypes):
            val = proto_alignment_matrix[i, j]
            color = 'black' if val > 0.5 else 'white'
            ax1.text(j, i, f'{val:.0%}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax1, label='Alignment Rate')

    # 1.2 Bar chart of mean alignment per prototype
    ax2 = axes[0, 1]
    mean_align = proto_alignment_matrix.mean(axis=0)
    std_align = proto_alignment_matrix.std(axis=0)

    colors = ['green' if proto_classes[i] == 0 else 'blue' for i in range(n_prototypes)]
    bars = ax2.bar(range(n_prototypes), mean_align, yerr=std_align, capsize=4, color=colors, alpha=0.7)
    ax2.axhline(y=0.7, color='green', linestyle='--', label='Good (70%)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Marginal (50%)')
    ax2.set_xlabel('Prototype Index')
    ax2.set_ylabel('Mean Alignment Rate')
    ax2.set_title('Mean Prototype Alignment Across Folds')
    ax2.set_xticks(range(n_prototypes))
    ax2.set_xticklabels([f'P{i}\n({"HC" if proto_classes[i]==0 else "MG"})' for i in range(n_prototypes)])
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    # 1.3 Comparison: HC vs MG prototypes
    ax3 = axes[1, 0]
    hc_protos = [i for i in range(n_prototypes) if proto_classes[i] == 0]
    mg_protos = [i for i in range(n_prototypes) if proto_classes[i] == 1]

    hc_aligns = [mean_align[i] for i in hc_protos]
    mg_aligns = [mean_align[i] for i in mg_protos]

    box_data = [hc_aligns, mg_aligns]
    bp = ax3.boxplot(box_data, labels=['HC Prototypes\n(0-4)', 'MG Prototypes\n(5-9)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('blue')
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Random (50%)')
    ax3.set_ylabel('Alignment Rate')
    ax3.set_title('Alignment by Prototype Class')
    ax3.legend()

    # 1.4 Scatter: Accuracy vs Prototype Alignment per fold
    ax4 = axes[1, 1]
    fold_accuracies = [fr['accuracy'] for fr in fold_results]
    fold_proto_align = proto_alignment_matrix.mean(axis=1)

    ax4.scatter(fold_proto_align, fold_accuracies, s=100, c=range(n_folds), cmap='viridis')
    for i in range(n_folds):
        ax4.annotate(f'Fold {i+1}', (fold_proto_align[i], fold_accuracies[i]),
                    xytext=(5, 5), textcoords='offset points')

    ax4.set_xlabel('Mean Prototype Alignment')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy vs Prototype Alignment')

    # Add correlation
    corr = np.corrcoef(fold_proto_align, fold_accuracies)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax4.transAxes,
            verticalalignment='top', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_alignment_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Neighbor distribution for each prototype
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))

    for proto_idx in range(n_prototypes):
        row = proto_idx // 5
        col = proto_idx % 5
        ax = axes[row, col]

        # Stack neighbor labels across folds
        hc_counts = []
        mg_counts = []
        for fr in fold_results:
            for p in fr['per_prototype_alignment']:
                if p['prototype_idx'] == proto_idx:
                    hc_counts.append(p['n_hc_neighbors'])
                    mg_counts.append(p['n_mg_neighbors'])

        mean_hc = np.mean(hc_counts)
        mean_mg = np.mean(mg_counts)

        proto_class = 'HC' if proto_classes[proto_idx] == 0 else 'MG'

        ax.bar(['HC', 'MG'], [mean_hc, mean_mg],
               color=['green' if proto_class == 'HC' else 'lightgreen',
                      'blue' if proto_class == 'MG' else 'lightblue'])
        ax.set_title(f'Proto {proto_idx} ({proto_class})\nAlign: {mean_align[proto_idx]:.0%}')
        ax.set_ylabel('Mean # Neighbors')
        ax.set_ylim(0, K_NEIGHBORS + 1)

        # Highlight which should be dominant
        expected = 'HC' if proto_class == 'HC' else 'MG'
        ax.axhline(y=K_NEIGHBORS * 0.5, color='red', linestyle='--', alpha=0.5)

    plt.suptitle(f'Neighbor Class Distribution per Prototype (k={K_NEIGHBORS})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_neighbor_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to: {output_dir}/")


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/ccece/prototype_alignment_investigation/{timestamp}"

    summary = run_investigation(output_dir, verbose=True)
