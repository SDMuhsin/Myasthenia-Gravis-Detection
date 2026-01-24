"""
Prototype Alignment Investigation: Phase 3 - Class Imbalance Analysis

Key insight: The alignment metric must be corrected for class imbalance.
- Dataset: 38.3% HC, 61.7% MG
- Random HC prototype alignment: ~38% (expected)
- Random MG prototype alignment: ~62% (expected)

If observed alignment matches these baselines, prototypes are not better than random.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def analyze_class_imbalance_effect(results_path: str, output_dir: str):
    """
    Analyze alignment relative to class imbalance baseline.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("PHASE 3: CLASS IMBALANCE ANALYSIS")
    print("="*70)

    # Dataset class distribution
    # From CONTEXT.md: 510 HC (38.3%), 821 MG (61.7%)
    hc_ratio = 0.383
    mg_ratio = 0.617

    print(f"\nDataset class distribution:")
    print(f"  HC: {hc_ratio:.1%}")
    print(f"  MG: {mg_ratio:.1%}")

    print(f"\nRandom baseline alignment:")
    print(f"  HC prototype (random): {hc_ratio:.1%} alignment")
    print(f"  MG prototype (random): {mg_ratio:.1%} alignment")

    # Analyze each prototype's alignment vs random baseline
    all_analysis = []

    for fold_result in results['per_fold_results']:
        fold = fold_result['fold']
        proto_alignment = fold_result['per_prototype_alignment']

        fold_analysis = {
            'fold': fold,
            'hc_prototypes': [],
            'mg_prototypes': [],
        }

        for p in proto_alignment:
            proto_idx = p['prototype_idx']
            proto_class = p['prototype_class']
            observed_align = p['alignment_rate']

            if proto_class == 0:  # HC
                expected_align = hc_ratio
                fold_analysis['hc_prototypes'].append({
                    'proto_idx': proto_idx,
                    'observed': observed_align,
                    'expected': expected_align,
                    'vs_random': observed_align - expected_align,
                })
            else:  # MG
                expected_align = mg_ratio
                fold_analysis['mg_prototypes'].append({
                    'proto_idx': proto_idx,
                    'observed': observed_align,
                    'expected': expected_align,
                    'vs_random': observed_align - expected_align,
                })

        all_analysis.append(fold_analysis)

    # Aggregate across folds
    hc_observed = []
    hc_vs_random = []
    mg_observed = []
    mg_vs_random = []

    for fa in all_analysis:
        for p in fa['hc_prototypes']:
            hc_observed.append(p['observed'])
            hc_vs_random.append(p['vs_random'])
        for p in fa['mg_prototypes']:
            mg_observed.append(p['observed'])
            mg_vs_random.append(p['vs_random'])

    print("\n" + "="*70)
    print("OBSERVED VS RANDOM BASELINE")
    print("="*70)

    print(f"\nHC Prototypes (n={len(hc_observed)}):")
    print(f"  Mean observed alignment: {np.mean(hc_observed):.1%}")
    print(f"  Expected random alignment: {hc_ratio:.1%}")
    print(f"  Improvement over random: {np.mean(hc_vs_random):.1%}")
    print(f"  Standard deviation: {np.std(hc_vs_random):.1%}")

    # Statistical test: is improvement significant?
    hc_improvement = np.mean(hc_vs_random)
    hc_se = np.std(hc_vs_random) / np.sqrt(len(hc_vs_random))
    hc_z = hc_improvement / hc_se if hc_se > 0 else 0

    print(f"  Z-score: {hc_z:.2f} {'(significant)' if abs(hc_z) > 1.96 else '(not significant)'}")

    print(f"\nMG Prototypes (n={len(mg_observed)}):")
    print(f"  Mean observed alignment: {np.mean(mg_observed):.1%}")
    print(f"  Expected random alignment: {mg_ratio:.1%}")
    print(f"  Improvement over random: {np.mean(mg_vs_random):.1%}")
    print(f"  Standard deviation: {np.std(mg_vs_random):.1%}")

    mg_improvement = np.mean(mg_vs_random)
    mg_se = np.std(mg_vs_random) / np.sqrt(len(mg_vs_random))
    mg_z = mg_improvement / mg_se if mg_se > 0 else 0

    print(f"  Z-score: {mg_z:.2f} {'(significant)' if abs(mg_z) > 1.96 else '(not significant)'}")

    # Overall assessment
    overall_observed = np.concatenate([hc_observed, mg_observed])
    overall_expected = np.concatenate([np.full(len(hc_observed), hc_ratio),
                                       np.full(len(mg_observed), mg_ratio)])
    overall_vs_random = overall_observed - overall_expected

    print(f"\nOVERALL (all prototypes):")
    print(f"  Mean observed alignment: {np.mean(overall_observed):.1%}")
    print(f"  Mean expected (random) alignment: {np.mean(overall_expected):.1%}")
    print(f"  Mean improvement over random: {np.mean(overall_vs_random):.1%}")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    print(f"""
1. OBSERVED ALIGNMENT: {np.mean(overall_observed):.1%}
2. RANDOM BASELINE (class imbalance adjusted): {np.mean(overall_expected):.1%}
3. IMPROVEMENT OVER RANDOM: {np.mean(overall_vs_random):.1%}

INTERPRETATION:
- The raw 49.6% alignment seemed problematic
- But the class-imbalance adjusted random baseline is {np.mean(overall_expected):.1%}
- So actual improvement over random is only {np.mean(overall_vs_random):.1%}

HC prototypes: observed {np.mean(hc_observed):.1%} vs random {hc_ratio:.1%} = +{np.mean(hc_vs_random):.1%}
MG prototypes: observed {np.mean(mg_observed):.1%} vs random {mg_ratio:.1%} = +{np.mean(mg_vs_random):.1%}

CONCLUSION:
Prototypes are barely better than random at attracting same-class neighbors.
The alignment problem is REAL, not an artifact of class imbalance.
""")

    # Generate visualization
    visualize_vs_random(hc_observed, mg_observed, hc_ratio, mg_ratio, output_dir)

    # Save analysis
    output = {
        'class_distribution': {'hc_ratio': hc_ratio, 'mg_ratio': mg_ratio},
        'hc_analysis': {
            'mean_observed': float(np.mean(hc_observed)),
            'expected_random': hc_ratio,
            'mean_improvement': float(np.mean(hc_vs_random)),
            'z_score': float(hc_z),
        },
        'mg_analysis': {
            'mean_observed': float(np.mean(mg_observed)),
            'expected_random': mg_ratio,
            'mean_improvement': float(np.mean(mg_vs_random)),
            'z_score': float(mg_z),
        },
        'overall': {
            'mean_observed': float(np.mean(overall_observed)),
            'mean_expected': float(np.mean(overall_expected)),
            'mean_improvement': float(np.mean(overall_vs_random)),
        },
    }

    output_path = os.path.join(output_dir, 'phase3_imbalance_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nAnalysis saved to: {output_path}")

    return output


def visualize_vs_random(hc_observed, mg_observed, hc_baseline, mg_baseline, output_dir):
    """Visualize alignment vs random baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Distribution of alignments by class
    ax1 = axes[0]

    positions = [1, 2]
    bp = ax1.boxplot([hc_observed, mg_observed], positions=positions, widths=0.6, patch_artist=True)

    colors = ['green', 'blue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add baseline lines
    ax1.hlines(hc_baseline, 0.5, 1.5, colors='green', linestyles='--', label=f'HC random ({hc_baseline:.0%})')
    ax1.hlines(mg_baseline, 1.5, 2.5, colors='blue', linestyles='--', label=f'MG random ({mg_baseline:.0%})')

    ax1.set_xticks(positions)
    ax1.set_xticklabels(['HC Prototypes', 'MG Prototypes'])
    ax1.set_ylabel('Alignment Rate')
    ax1.set_title('Prototype Alignment vs Random Baseline')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Add text showing improvement
    hc_imp = np.mean(hc_observed) - hc_baseline
    mg_imp = np.mean(mg_observed) - mg_baseline
    ax1.text(1, np.mean(hc_observed) + 0.05, f'+{hc_imp:.0%}', ha='center', fontsize=10, color='green')
    ax1.text(2, np.mean(mg_observed) + 0.05, f'+{mg_imp:.0%}', ha='center', fontsize=10, color='blue')

    # 2. Improvement over random
    ax2 = axes[1]

    hc_vs_random = np.array(hc_observed) - hc_baseline
    mg_vs_random = np.array(mg_observed) - mg_baseline

    ax2.hist(hc_vs_random, bins=10, alpha=0.6, color='green', label='HC prototypes')
    ax2.hist(mg_vs_random, bins=10, alpha=0.6, color='blue', label='MG prototypes')

    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Random baseline')
    ax2.axvline(x=np.mean(hc_vs_random), color='green', linestyle='-', linewidth=2)
    ax2.axvline(x=np.mean(mg_vs_random), color='blue', linestyle='-', linewidth=2)

    ax2.set_xlabel('Improvement over Random Baseline')
    ax2.set_ylabel('Count')
    ax2.set_title('Alignment Improvement Distribution')
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'phase3_imbalance_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {save_path}")


if __name__ == '__main__':
    import glob
    pattern = "results/ccece/prototype_alignment_investigation/*/investigation_results.json"
    files = sorted(glob.glob(pattern))

    if not files:
        print("No Phase 1 results found. Run prototype_alignment_investigation.py first.")
        sys.exit(1)

    results_path = files[-1]
    print(f"Using Phase 1 results: {results_path}")

    output_dir = os.path.dirname(results_path)
    analyze_class_imbalance_effect(results_path, output_dir)
