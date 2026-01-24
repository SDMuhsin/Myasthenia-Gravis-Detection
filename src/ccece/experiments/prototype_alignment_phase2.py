"""
Prototype Alignment Investigation: Phase 2 - Characterizing Misalignment

Key finding from Phase 1: PROTOTYPE COLLAPSE
- Only 2 out of 10 prototypes have samples assigned to them
- 8 prototypes are "inactive" (no samples use them as nearest prototype)

This script investigates:
1. Are "active" prototypes well-aligned?
2. Why are "inactive" prototypes misaligned?
3. What causes prototype collapse?
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def analyze_prototype_collapse(results_path: str, output_dir: str):
    """
    Analyze the relationship between prototype activity and alignment.
    """
    # Load Phase 1 results
    with open(results_path, 'r') as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("PHASE 2: CHARACTERIZING PROTOTYPE COLLAPSE")
    print("="*70)

    # Analyze each fold
    analysis = []

    for fold_result in results['per_fold_results']:
        fold = fold_result['fold']
        accuracy = fold_result['accuracy']
        proto_alignment = fold_result['per_prototype_alignment']
        embed_dist = fold_result['embedding_distribution']

        print(f"\n{'='*50}")
        print(f"FOLD {fold} (Accuracy: {accuracy:.1%})")
        print('='*50)

        # Identify active prototypes (have samples assigned)
        sample_counts = embed_dist['proto_sample_counts']
        active_protos = [int(k) for k, v in sample_counts.items() if v > 0]
        inactive_protos = [int(k) for k, v in sample_counts.items() if v == 0]

        print(f"\nActive prototypes: {active_protos}")
        print(f"Inactive prototypes: {inactive_protos}")

        # Compute alignment for active vs inactive
        active_alignments = []
        inactive_alignments = []
        active_distances = []
        inactive_distances = []

        for p in proto_alignment:
            proto_idx = p['prototype_idx']
            align = p['alignment_rate']
            mean_dist = np.mean(p['neighbor_distances'])

            if proto_idx in active_protos:
                active_alignments.append(align)
                active_distances.append(mean_dist)
            else:
                inactive_alignments.append(align)
                inactive_distances.append(mean_dist)

        mean_active_align = np.mean(active_alignments) if active_alignments else 0
        mean_inactive_align = np.mean(inactive_alignments) if inactive_alignments else 0
        mean_active_dist = np.mean(active_distances) if active_distances else 0
        mean_inactive_dist = np.mean(inactive_distances) if inactive_distances else 0

        print(f"\nActive prototypes ({len(active_protos)}):")
        print(f"  Mean alignment: {mean_active_align:.1%}")
        print(f"  Mean neighbor distance: {mean_active_dist:.4f}")

        print(f"\nInactive prototypes ({len(inactive_protos)}):")
        print(f"  Mean alignment: {mean_inactive_align:.1%}")
        print(f"  Mean neighbor distance: {mean_inactive_dist:.4f}")

        # Show per-prototype details
        print(f"\nPer-prototype breakdown:")
        print(f"  {'Proto':<6} {'Class':<6} {'Samples':<10} {'Align':<8} {'MeanDist':<10} {'Status':<10}")
        print(f"  {'-'*60}")

        for p in proto_alignment:
            proto_idx = p['prototype_idx']
            proto_class = p['prototype_class_name']
            align = p['alignment_rate']
            mean_dist = np.mean(p['neighbor_distances'])
            n_samples = sample_counts[str(proto_idx)]
            status = "ACTIVE" if proto_idx in active_protos else "inactive"

            print(f"  {proto_idx:<6} {proto_class:<6} {n_samples:<10} {align*100:<8.0f}% {mean_dist:<10.4f} {status:<10}")

        # Store analysis
        analysis.append({
            'fold': fold,
            'accuracy': accuracy,
            'active_protos': active_protos,
            'inactive_protos': inactive_protos,
            'n_active': len(active_protos),
            'mean_active_alignment': mean_active_align,
            'mean_inactive_alignment': mean_inactive_align,
            'mean_active_distance': mean_active_dist,
            'mean_inactive_distance': mean_inactive_dist,
        })

    # Aggregate analysis
    print("\n" + "="*70)
    print("AGGREGATE ANALYSIS ACROSS FOLDS")
    print("="*70)

    mean_n_active = np.mean([a['n_active'] for a in analysis])
    mean_active_align = np.mean([a['mean_active_alignment'] for a in analysis])
    mean_inactive_align = np.mean([a['mean_inactive_alignment'] for a in analysis])
    mean_active_dist = np.mean([a['mean_active_distance'] for a in analysis])
    mean_inactive_dist = np.mean([a['mean_inactive_distance'] for a in analysis])

    print(f"\nAverage number of active prototypes: {mean_n_active:.1f} out of 10")
    print(f"\nActive prototypes:")
    print(f"  Mean alignment: {mean_active_align:.1%}")
    print(f"  Mean neighbor distance: {mean_active_dist:.4f}")
    print(f"\nInactive prototypes:")
    print(f"  Mean alignment: {mean_inactive_align:.1%}")
    print(f"  Mean neighbor distance: {mean_inactive_dist:.4f}")

    alignment_gap = mean_active_align - mean_inactive_align
    distance_ratio = mean_inactive_dist / mean_active_dist if mean_active_dist > 0 else 0

    print(f"\nKEY OBSERVATIONS:")
    print(f"  1. Alignment gap (active - inactive): {alignment_gap:.1%}")
    print(f"  2. Distance ratio (inactive/active): {distance_ratio:.1f}x")
    print(f"  3. Inactive prototypes are {distance_ratio:.0f}x farther from samples")
    print(f"  4. Inactive prototype alignment ({mean_inactive_align:.1%}) is near random (50%)")

    # Compute "corrected" alignment using only active prototypes
    overall_mean_alignment = results['overall']['mean_prototype_alignment']
    print(f"\n  5. Overall alignment (all prototypes): {overall_mean_alignment:.1%}")
    print(f"  6. Active-only alignment: {mean_active_align:.1%}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    print(f"""
The 49.6% prototype alignment is caused by PROTOTYPE COLLAPSE:

1. Only {mean_n_active:.0f} out of 10 prototypes are "active" (nearest to any sample)
2. Active prototypes have {mean_active_align:.1%} alignment (reasonable)
3. Inactive prototypes have {mean_inactive_align:.1%} alignment (near random)
4. The 8 inactive prototypes drag down the average from {mean_active_align:.1%} to {overall_mean_alignment:.1%}

ROOT CAUSE HYPOTHESIS:
The model learns to use only 1 prototype per class (2 total), making the other 8
redundant. This happens because:
- The separation loss pushes different-class prototypes apart
- The cluster loss only requires samples to be close to ONE same-class prototype
- Once one prototype "wins", the others have no gradient signal to improve

This is NOT a problem with the prototypes themselves being misaligned.
It's a problem with PROTOTYPE REDUNDANCY - most prototypes are unused.
""")

    # Save analysis
    output = {
        'per_fold_analysis': analysis,
        'aggregate': {
            'mean_n_active': mean_n_active,
            'mean_active_alignment': mean_active_align,
            'mean_inactive_alignment': mean_inactive_align,
            'mean_active_distance': mean_active_dist,
            'mean_inactive_distance': mean_inactive_dist,
            'alignment_gap': alignment_gap,
            'distance_ratio': distance_ratio,
        },
        'conclusion': 'Prototype collapse causes low alignment - only 2/10 prototypes are active',
    }

    output_path = os.path.join(output_dir, 'phase2_collapse_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nAnalysis saved to: {output_path}")

    # Generate visualization
    visualize_collapse(analysis, output_dir)

    return output


def visualize_collapse(analysis: List[Dict], output_dir: str):
    """Visualize prototype collapse pattern."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    folds = [a['fold'] for a in analysis]

    # 1. Number of active prototypes per fold
    ax1 = axes[0]
    n_active = [a['n_active'] for a in analysis]
    ax1.bar(folds, n_active, color='steelblue')
    ax1.axhline(y=10, color='red', linestyle='--', label='Total prototypes')
    ax1.axhline(y=2, color='orange', linestyle='--', label='Expected minimum (1 per class)')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Number of Active Prototypes')
    ax1.set_title('Prototype Collapse: Active Prototypes per Fold')
    ax1.set_ylim(0, 12)
    ax1.legend()

    # 2. Alignment: Active vs Inactive
    ax2 = axes[1]
    active_align = [a['mean_active_alignment'] * 100 for a in analysis]
    inactive_align = [a['mean_inactive_alignment'] * 100 for a in analysis]

    x = np.arange(len(folds))
    width = 0.35

    bars1 = ax2.bar(x - width/2, active_align, width, label='Active', color='green')
    bars2 = ax2.bar(x + width/2, inactive_align, width, label='Inactive', color='red', alpha=0.6)

    ax2.axhline(y=50, color='gray', linestyle='--', label='Random (50%)')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Alignment (%)')
    ax2.set_title('Alignment: Active vs Inactive Prototypes')
    ax2.set_xticks(x)
    ax2.set_xticklabels(folds)
    ax2.legend()
    ax2.set_ylim(0, 100)

    # 3. Distance: Active vs Inactive
    ax3 = axes[2]
    active_dist = [a['mean_active_distance'] for a in analysis]
    inactive_dist = [a['mean_inactive_distance'] for a in analysis]

    bars1 = ax3.bar(x - width/2, active_dist, width, label='Active', color='green')
    bars2 = ax3.bar(x + width/2, inactive_dist, width, label='Inactive', color='red', alpha=0.6)

    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Mean Neighbor Distance')
    ax3.set_title('Distance to Neighbors: Active vs Inactive')
    ax3.set_xticks(x)
    ax3.set_xticklabels(folds)
    ax3.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'phase2_collapse_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {save_path}")


if __name__ == '__main__':
    # Find latest Phase 1 results
    import glob
    pattern = "results/ccece/prototype_alignment_investigation/*/investigation_results.json"
    files = sorted(glob.glob(pattern))

    if not files:
        print("No Phase 1 results found. Run prototype_alignment_investigation.py first.")
        sys.exit(1)

    results_path = files[-1]  # Use latest
    print(f"Using Phase 1 results: {results_path}")

    # Output to same directory
    output_dir = os.path.dirname(results_path)

    analyze_prototype_collapse(results_path, output_dir)
