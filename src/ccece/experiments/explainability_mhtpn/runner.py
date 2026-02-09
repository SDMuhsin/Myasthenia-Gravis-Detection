"""
Main Runner for MHTPN Explainability Analysis

Orchestrates all 4 components:
1. Trajectory Prototype Analysis
2. Per-Segment Decision Analysis
3. Head Velocity Diversity Analysis
4. Sample-Level Case Studies
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import create_data_loaders
from ccece.models.multi_head_trajectory_proto_net import MultiHeadTrajectoryProtoNet
from ccece.experiments.multi_head_trajectory_experiment import MHTConfig, MHTTrainer

from .trajectory_analysis import analyze_trajectory_prototypes
from .segment_analysis import analyze_per_segment_decisions, analyze_per_head_segment_patterns
from .velocity_diversity import analyze_velocity_diversity, compute_per_head_velocity_analysis
from .case_studies import analyze_case_studies
from .figure_generator import generate_all_figures


RANDOM_SEED = 42


def convert_to_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(v) for v in obj)
    return obj


def save_json(data: Dict, path: str):
    """Save dictionary as JSON file, handling numpy types."""
    serializable = convert_to_json_serializable(data)
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def train_model_for_explainability(
    config: MHTConfig,
    items: List,
    seq_len: int,
    input_dim: int,
    device: torch.device,
    verbose: bool = True,
) -> tuple:
    """
    Train a single MHTPN model for explainability analysis.

    Uses the full dataset with a train/val split (not cross-validation)
    to get a single model for analysis.

    Returns:
        (model, val_loader, metrics)
    """
    set_all_seeds(RANDOM_SEED)

    X, y, patient_ids = extract_arrays(items)

    # Use stratified group split: 80% train, 20% val
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Take the first fold
    train_idx, val_idx = next(cv.split(X, y, patient_ids))

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    train_labels = np.array([item['label'] for item in train_items])

    if verbose:
        print(f"Training set: {len(train_items)} samples")
        print(f"Validation set: {len(val_items)} samples")

    # Create data loaders
    train_loader, val_loader, scaler = create_data_loaders(
        train_items, val_items, seq_len, config.batch_size
    )

    # Create model
    model = MultiHeadTrajectoryProtoNet(
        input_dim=input_dim,
        num_classes=2,
        seq_len=seq_len,
        latent_dim=config.latent_dim,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        n_segments=config.n_segments,
        encoder_hidden=config.encoder_hidden,
        encoder_layers=config.encoder_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )

    if verbose:
        print(f"Model parameters: {model.count_parameters():,}")

    # Train
    trainer = MHTTrainer(model, config, device)
    train_result = trainer.train(train_loader, val_loader, train_labels, verbose=verbose)

    # Evaluate
    metrics = trainer.evaluate(val_loader)

    if verbose:
        print(f"\nValidation Accuracy: {metrics.accuracy:.1%}")

    return model, val_loader, metrics


def run_explainability(
    output_dir: str,
    verbose: bool = True,
    components: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full MHTPN explainability analysis.

    Args:
        output_dir: Directory to save results
        verbose: Whether to print progress
        components: Optional list of components to run. If None, runs all.
                   Options: 'trajectory', 'segment', 'diversity', 'case_studies'

    Returns:
        Dict with all analysis results
    """
    set_all_seeds(RANDOM_SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'quantitative'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 70)
        print("MHTPN EXPLAINABILITY ANALYSIS")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Output: {output_dir}")

    # Determine which components to run
    all_components = ['trajectory', 'segment', 'diversity', 'case_studies']
    if components is None:
        components = all_components
    else:
        invalid = set(components) - set(all_components)
        if invalid:
            raise ValueError(f"Invalid components: {invalid}")

    if verbose:
        print(f"Components to run: {components}")

    # Load data
    if verbose:
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)

    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)
    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]

    if verbose:
        print(f"Data: {len(items)} samples, seq_len={seq_len}, input_dim={input_dim}")

    # Create config
    config = MHTConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=32,
        n_segments=8,
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=20,
        cluster_loss_weight=0.3,
        separation_loss_weight=0.1,
        diversity_loss_weight=0.05,
        n_folds=5,
    )

    # Train model
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)

    model, val_loader, eval_metrics = train_model_for_explainability(
        config, items, seq_len, input_dim, device, verbose=verbose
    )

    # Initialize results
    results = {
        'metadata': {
            'timestamp': timestamp,
            'random_seed': RANDOM_SEED,
            'device': str(device),
            'seq_len': seq_len,
            'input_dim': input_dim,
            'n_samples': len(items),
            'validation_accuracy': float(eval_metrics.accuracy),
            'config': {
                'latent_dim': config.latent_dim,
                'n_heads': config.n_heads,
                'head_dim': config.head_dim,
                'n_segments': config.n_segments,
            },
        },
        'components': {},
    }

    # =========================================================================
    # Component 1: Trajectory Prototype Analysis
    # =========================================================================
    if 'trajectory' in components:
        if verbose:
            print("\n" + "=" * 60)
            print("COMPONENT 1: TRAJECTORY PROTOTYPE ANALYSIS")
            print("=" * 60)

        trajectory_results = analyze_trajectory_prototypes(model, device)
        results['components']['trajectory'] = trajectory_results

        # Save quantitative output
        trajectory_path = os.path.join(output_dir, 'quantitative', 'trajectory_analysis.json')
        save_json(trajectory_results, trajectory_path)

        if verbose:
            print(f"Mean velocity norm: {trajectory_results['summary']['mean_velocity_norm']:.4f}")
            print(f"All heads have motion: {trajectory_results['summary']['all_heads_have_motion']}")
            print(f"Mean origin separation: {trajectory_results['summary']['mean_origin_separation']:.4f}")

    # =========================================================================
    # Component 2: Per-Segment Decision Analysis
    # =========================================================================
    if 'segment' in components:
        if verbose:
            print("\n" + "=" * 60)
            print("COMPONENT 2: PER-SEGMENT DECISION ANALYSIS")
            print("=" * 60)

        segment_results = analyze_per_segment_decisions(model, val_loader, device)
        per_head_segment_results = analyze_per_head_segment_patterns(model, val_loader, device)

        results['components']['segment'] = segment_results
        results['components']['per_head_segment'] = per_head_segment_results

        # Save quantitative output
        segment_path = os.path.join(output_dir, 'quantitative', 'segment_analysis.json')
        save_json(segment_results, segment_path)

        per_head_path = os.path.join(output_dir, 'quantitative', 'per_head_segment_analysis.json')
        save_json(per_head_segment_results, per_head_path)

        if verbose:
            print(f"Temporal pattern pass: {segment_results['temporal_pattern_pass']}")
            print(f"Late minus early: {segment_results['late_minus_early']:.4f}")
            print(f"Most discriminative segment: {segment_results['most_discriminative_segment']}")
            print(f"Heads with late > early: {per_head_segment_results['aggregate']['heads_with_late_gt_early']}/5")

    # =========================================================================
    # Component 3: Head Velocity Diversity Analysis
    # =========================================================================
    if 'diversity' in components:
        if verbose:
            print("\n" + "=" * 60)
            print("COMPONENT 3: HEAD VELOCITY DIVERSITY ANALYSIS")
            print("=" * 60)

        diversity_results = analyze_velocity_diversity(model, device)
        per_head_velocity_results = compute_per_head_velocity_analysis(model, device)

        results['components']['diversity'] = diversity_results
        results['components']['per_head_velocity'] = per_head_velocity_results

        # Save quantitative output
        diversity_path = os.path.join(output_dir, 'quantitative', 'velocity_diversity.json')
        save_json(diversity_results, diversity_path)

        per_head_vel_path = os.path.join(output_dir, 'quantitative', 'per_head_velocity.json')
        save_json(per_head_velocity_results, per_head_vel_path)

        if verbose:
            print(f"Diversity score: {diversity_results['diversity_score']:.4f}")
            print(f"Mean off-diagonal similarity: {diversity_results['mean_off_diagonal_similarity']:.4f}")
            print(f"Min velocity norm: {diversity_results['min_velocity_norm']:.4f}")
            print(f"Interpretation: {diversity_results['interpretation']}")

    # =========================================================================
    # Component 4: Sample-Level Case Studies
    # =========================================================================
    if 'case_studies' in components:
        if verbose:
            print("\n" + "=" * 60)
            print("COMPONENT 4: SAMPLE-LEVEL CASE STUDIES")
            print("=" * 60)

        case_study_results = analyze_case_studies(model, val_loader, device, n_per_category=5)
        results['components']['case_studies'] = case_study_results

        # Save quantitative output
        case_study_path = os.path.join(output_dir, 'quantitative', 'case_studies.json')
        save_json(case_study_results, case_study_path)

        if verbose:
            print(f"Total samples analyzed: {case_study_results['summary']['total_samples_analyzed']}")
            print(f"MG mean slope: {case_study_results['summary']['mg_mean_slope']}")
            print(f"HC mean slope: {case_study_results['summary']['hc_mean_slope']}")
            print(f"Slope test significant: {case_study_results['summary']['slope_test_significant']}")

    # =========================================================================
    # Generate Figures
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("GENERATING FIGURES")
        print("=" * 60)

    # Get results for each component (use empty dicts if not run)
    trajectory_results = results['components'].get('trajectory', {})
    segment_results = results['components'].get('segment', {})
    per_head_segment_results = results['components'].get('per_head_segment', {})
    diversity_results = results['components'].get('diversity', {})
    case_study_results = results['components'].get('case_studies', {})

    if all(key in components for key in all_components):
        generated_figures = generate_all_figures(
            trajectory_results,
            segment_results,
            per_head_segment_results,
            diversity_results,
            case_study_results,
            output_dir,
        )
        results['generated_figures'] = {
            comp: [os.path.basename(f) for f in figs]
            for comp, figs in generated_figures.items()
        }

        if verbose:
            total_figs = sum(len(figs) for figs in generated_figures.values())
            print(f"Generated {total_figs} figures")

    # =========================================================================
    # Success Criteria Summary
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("SUCCESS CRITERIA SUMMARY")
        print("=" * 60)

    success_summary = {}

    if 'trajectory' in results['components']:
        traj = results['components']['trajectory']['success_criteria']
        success_summary['trajectory'] = {
            k: v['passed'] for k, v in traj.items()
        }
        if verbose:
            print("\nTrajectory Analysis:")
            for name, data in traj.items():
                status = "PASS" if data['passed'] else "FAIL"
                print(f"  {name}: {status} ({data['threshold']}, value={data['value']:.4f})")

    if 'segment' in results['components']:
        seg = results['components']['segment']['success_criteria']
        success_summary['segment'] = {
            k: v['passed'] for k, v in seg.items()
        }
        if verbose:
            print("\nSegment Analysis:")
            for name, data in seg.items():
                status = "PASS" if data['passed'] else "FAIL"
                print(f"  {name}: {status}")

    if 'diversity' in results['components']:
        div = results['components']['diversity']['success_criteria']
        success_summary['diversity'] = {
            k: v['passed'] for k, v in div.items()
        }
        if verbose:
            print("\nDiversity Analysis:")
            for name, data in div.items():
                status = "PASS" if data['passed'] else "FAIL"
                print(f"  {name}: {status} ({data['threshold']}, value={data['value']:.4f})")

    if 'case_studies' in results['components']:
        cases = results['components']['case_studies']['success_criteria']
        success_summary['case_studies'] = {
            k: v['passed'] for k, v in cases.items()
        }
        if verbose:
            print("\nCase Studies:")
            for name, data in cases.items():
                status = "PASS" if data['passed'] else "FAIL"
                print(f"  {name}: {status}")

    results['success_summary'] = success_summary

    # Overall pass count
    all_passed = []
    for comp_results in success_summary.values():
        all_passed.extend(comp_results.values())

    n_passed = sum(all_passed)
    n_total = len(all_passed)
    results['overall'] = {
        'criteria_passed': n_passed,
        'criteria_total': n_total,
        'all_passed': n_passed == n_total,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"OVERALL: {n_passed}/{n_total} criteria passed")
        if n_passed == n_total:
            print("STATUS: ALL SUCCESS CRITERIA MET")
        else:
            print("STATUS: SOME CRITERIA FAILED")
        print("=" * 60)

    # Save full results
    full_results_path = os.path.join(output_dir, 'full_results.json')
    save_json(results, full_results_path)

    # Generate summary text file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MHTPN Explainability Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Validation Accuracy: {eval_metrics.accuracy:.1%}\n\n")

        f.write("Success Criteria:\n")
        for comp, criteria in success_summary.items():
            f.write(f"\n{comp}:\n")
            for name, passed in criteria.items():
                status = "PASS" if passed else "FAIL"
                f.write(f"  {name}: {status}\n")

        f.write(f"\nOverall: {n_passed}/{n_total} criteria passed\n")

    if verbose:
        print(f"\nResults saved to: {output_dir}")

    return results


def main():
    """Main entry point."""
    output_dir = "results/ccece/explainability_mhtpn"
    run_explainability(output_dir, verbose=True)


if __name__ == '__main__':
    main()
