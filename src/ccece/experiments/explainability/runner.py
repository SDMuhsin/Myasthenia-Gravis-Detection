#!/usr/bin/env python3
"""
CCECE Paper: Experiment 03 - Explainability Analysis Runner

Main orchestrator for all 5 explainability components:
1. Prototype Analysis
2. Temporal Saliency Analysis
3. Feature Importance Analysis
4. Head Specialization Analysis
5. Case Study Analysis

Usage:
    python src/ccece/experiments/explainability/runner.py
    python src/ccece/experiments/explainability/runner.py --component prototype
    python src/ccece/experiments/explainability/runner.py --component saliency
    python src/ccece/experiments/explainability/runner.py --component features
    python src/ccece/experiments/explainability/runner.py --component heads
    python src/ccece/experiments/explainability/runner.py --component cases
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import create_data_loaders, SequenceScaler, SaccadeDataset
from ccece.models.multi_head_proto_net import MultiHeadProtoNet

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = 42
RESULTS_BASE_DIR = 'results/ccece/explainability'

# Channel names (14 channels)
CHANNEL_NAMES = [
    'LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV',
    'LH_Velocity', 'RH_Velocity', 'LV_Velocity', 'RV_Velocity',
    'ErrorH_L', 'ErrorH_R', 'ErrorV_L', 'ErrorV_R'
]

# Channel categories
CHANNEL_CATEGORIES = {
    'position': ['LH', 'RH', 'LV', 'RV'],
    'target': ['TargetH', 'TargetV'],
    'velocity': ['LH_Velocity', 'RH_Velocity', 'LV_Velocity', 'RV_Velocity'],
    'error': ['ErrorH_L', 'ErrorH_R', 'ErrorV_L', 'ErrorV_R']
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExplainabilityConfig:
    """Configuration for explainability analysis."""
    # Model configuration (must match trained model)
    latent_dim: int = 64
    n_heads: int = 5
    head_dim: int = 64
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2

    # Training configuration for model retraining
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    cluster_loss_weight: float = 0.3
    separation_loss_weight: float = 0.1

    # Explainability settings
    n_saliency_samples: int = 100  # Number of samples for saliency analysis
    ig_steps: int = 50  # Integrated gradients steps
    perm_n_repeats: int = 10  # Permutation importance repeats
    n_case_studies_per_category: int = 5  # Case studies per category


# =============================================================================
# UTILITIES
# =============================================================================

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
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
    elif isinstance(obj, tuple):
        return [convert_to_serializable(v) for v in obj]
    return obj


def save_json(filepath: str, data: Dict):
    """Save data to JSON file with proper serialization."""
    with open(filepath, 'w') as f:
        json.dump(convert_to_serializable(data), f, indent=2)


def create_output_dirs(base_dir: str) -> Dict[str, str]:
    """Create output directory structure."""
    dirs = {
        'base': base_dir,
        'figures': os.path.join(base_dir, 'figures'),
        'quantitative': os.path.join(base_dir, 'quantitative'),
        'tables': os.path.join(base_dir, 'tables'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(
    train_items: List[Dict],
    val_items: List[Dict],
    config: ExplainabilityConfig,
    device: torch.device,
    seq_len: int,
    input_dim: int,
    verbose: bool = True
) -> MultiHeadProtoNet:
    """
    Train a MultiHeadProtoNet model on the given data.

    Returns the trained model.
    """
    from ccece.experiments.multi_head_proto_experiment import MultiHeadTrainer, MultiHeadConfig

    # Create data loaders
    train_loader, val_loader, scaler = create_data_loaders(
        train_items, val_items, seq_len, config.batch_size
    )

    # Create model
    model = MultiHeadProtoNet(
        input_dim=input_dim,
        num_classes=2,
        seq_len=seq_len,
        latent_dim=config.latent_dim,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        encoder_hidden=config.encoder_hidden,
        encoder_layers=config.encoder_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )

    # Create trainer config
    trainer_config = MultiHeadConfig(
        latent_dim=config.latent_dim,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        encoder_hidden=config.encoder_hidden,
        encoder_layers=config.encoder_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_patience=config.early_stopping_patience,
        cluster_loss_weight=config.cluster_loss_weight,
        separation_loss_weight=config.separation_loss_weight,
        per_head_ce_weight=0.0,
    )

    # Train
    train_labels = np.array([item['label'] for item in train_items])
    trainer = MultiHeadTrainer(model, trainer_config, device)
    trainer.train(train_loader, val_loader, train_labels, verbose=verbose)

    return model, scaler, train_loader, val_loader


# =============================================================================
# MAIN RUNNER
# =============================================================================

class ExplainabilityRunner:
    """Main runner for explainability analysis."""

    def __init__(
        self,
        config: ExplainabilityConfig,
        output_dir: str,
        device: torch.device,
        verbose: bool = True,
    ):
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.verbose = verbose
        self.dirs = create_output_dirs(output_dir)

        # Will be populated during run
        self.model = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.train_items = None
        self.val_items = None
        self.seq_len = None
        self.input_dim = None

        # Results storage
        self.results = {}

    def log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    def prepare_data(self):
        """Load and prepare data."""
        self.log("\n" + "="*60)
        self.log("LOADING AND PREPARING DATA")
        self.log("="*60)

        # Load data
        items = load_binary_dataset(verbose=self.verbose)
        items = preprocess_items(items)

        X, y, patient_ids = extract_arrays(items)
        self.seq_len = compute_target_seq_len(items)
        self.input_dim = items[0]['data'].shape[1]

        self.log(f"\nData: {len(items)} samples, seq_len={self.seq_len}, input_dim={self.input_dim}")

        # Use first fold for analysis (same split as training)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        train_idx, val_idx = next(iter(cv.split(X, y, patient_ids)))

        self.train_items = [items[i] for i in train_idx]
        self.val_items = [items[i] for i in val_idx]

        self.log(f"Train: {len(self.train_items)} samples")
        self.log(f"Val: {len(self.val_items)} samples")

    def train_model(self):
        """Train the model for analysis."""
        self.log("\n" + "="*60)
        self.log("TRAINING MODEL")
        self.log("="*60)

        self.model, self.scaler, self.train_loader, self.val_loader = train_model(
            self.train_items,
            self.val_items,
            self.config,
            self.device,
            self.seq_len,
            self.input_dim,
            verbose=self.verbose
        )

        # Store training embeddings for prototype analysis
        from ccece.experiments.multi_head_proto_experiment import store_training_embeddings
        store_training_embeddings(self.model, self.train_loader, self.device)

    def run_component(self, component: str):
        """Run a specific component of the analysis."""
        if component == 'prototype':
            return self.run_prototype_analysis()
        elif component == 'saliency':
            return self.run_saliency_analysis()
        elif component == 'features':
            return self.run_feature_importance()
        elif component == 'heads':
            return self.run_head_analysis()
        elif component == 'cases':
            return self.run_case_studies()
        else:
            raise ValueError(f"Unknown component: {component}")

    def run_prototype_analysis(self) -> Dict:
        """Run Component 1: Prototype Analysis."""
        self.log("\n" + "="*60)
        self.log("COMPONENT 1: PROTOTYPE ANALYSIS")
        self.log("="*60)

        from ccece.experiments.explainability.prototype_analysis import PrototypeAnalyzer

        analyzer = PrototypeAnalyzer(
            model=self.model,
            device=self.device,
            output_dir=self.dirs['figures'],
            quantitative_dir=self.dirs['quantitative'],
        )

        results = analyzer.run_analysis(self.train_loader)
        self.results['prototype_analysis'] = results

        self.log(f"\nPrototype Analysis Results:")
        self.log(f"  Separability ratio: {results['separability_ratio']:.3f}")
        self.log(f"  Inter-class distance: {results['inter_class_distance']:.3f}")
        self.log(f"  Intra-class distance: {results['intra_class_distance']:.3f}")

        return results

    def run_saliency_analysis(self) -> Dict:
        """Run Component 2: Temporal Saliency Analysis."""
        self.log("\n" + "="*60)
        self.log("COMPONENT 2: TEMPORAL SALIENCY ANALYSIS")
        self.log("="*60)

        from ccece.experiments.explainability.saliency import TemporalSaliencyAnalyzer

        analyzer = TemporalSaliencyAnalyzer(
            model=self.model,
            device=self.device,
            output_dir=self.dirs['figures'],
            quantitative_dir=self.dirs['quantitative'],
            ig_steps=self.config.ig_steps,
            n_samples=self.config.n_saliency_samples,
        )

        results = analyzer.run_analysis(self.val_loader, CHANNEL_NAMES)
        self.results['temporal_saliency'] = results

        self.log(f"\nTemporal Saliency Results:")
        self.log(f"  MG peak location: {results['mg']['peak_location_mean']:.3f}")
        self.log(f"  HC peak location: {results['hc']['peak_location_mean']:.3f}")
        early_late = results.get('statistical_tests', {}).get('early_vs_late_ratio', {})
        if early_late:
            self.log(f"  Early/late ratio significant: {early_late.get('significant', False)}")

        return results

    def run_feature_importance(self) -> Dict:
        """Run Component 3: Feature Importance Analysis."""
        self.log("\n" + "="*60)
        self.log("COMPONENT 3: FEATURE IMPORTANCE ANALYSIS")
        self.log("="*60)

        from ccece.experiments.explainability.feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer(
            model=self.model,
            device=self.device,
            output_dir=self.dirs['figures'],
            quantitative_dir=self.dirs['quantitative'],
            n_repeats=self.config.perm_n_repeats,
        )

        results = analyzer.run_analysis(
            self.val_loader,
            CHANNEL_NAMES,
            CHANNEL_CATEGORIES
        )
        self.results['feature_importance'] = results

        self.log(f"\nFeature Importance Results:")
        self.log(f"  Top 3 features: {[r['channel'] for r in results['ranking'][:3]]}")
        self.log(f"  Velocity vs Position: p={results['velocity_vs_position']['p_value']:.4f}")

        return results

    def run_head_analysis(self) -> Dict:
        """Run Component 4: Head Specialization Analysis."""
        self.log("\n" + "="*60)
        self.log("COMPONENT 4: HEAD SPECIALIZATION ANALYSIS")
        self.log("="*60)

        from ccece.experiments.explainability.head_analysis import HeadSpecializationAnalyzer

        analyzer = HeadSpecializationAnalyzer(
            model=self.model,
            device=self.device,
            output_dir=self.dirs['figures'],
            quantitative_dir=self.dirs['quantitative'],
        )

        results = analyzer.run_analysis(self.val_loader, CHANNEL_NAMES)
        self.results['head_analysis'] = results

        self.log(f"\nHead Analysis Results:")
        self.log(f"  Mean pairwise agreement: {results['mean_pairwise_agreement']:.3f}")
        head_accs = [f"{results['head_performance'][f'head_{i}']['accuracy']:.3f}" for i in range(self.config.n_heads)]
        self.log(f"  Per-head accuracies: {head_accs}")

        return results

    def run_case_studies(self) -> Dict:
        """Run Component 5: Case Study Analysis."""
        self.log("\n" + "="*60)
        self.log("COMPONENT 5: CASE STUDY ANALYSIS")
        self.log("="*60)

        from ccece.experiments.explainability.case_studies import CaseStudyAnalyzer

        analyzer = CaseStudyAnalyzer(
            model=self.model,
            device=self.device,
            output_dir=self.dirs['figures'],
            quantitative_dir=self.dirs['quantitative'],
            n_per_category=self.config.n_case_studies_per_category,
        )

        results = analyzer.run_analysis(
            self.val_loader,
            self.val_items,
            CHANNEL_NAMES
        )
        self.results['case_studies'] = results

        self.log(f"\nCase Study Results:")
        self.log(f"  High confidence correct: {len(results.get('high_confidence_correct_mg', []))} MG + {len(results.get('high_confidence_correct_hc', []))} HC")
        self.log(f"  Misclassified: {len(results.get('misclassified_mg_to_hc', []))} MG→HC + {len(results.get('misclassified_hc_to_mg', []))} HC→MG")

        return results

    def generate_report(self):
        """Generate final report and tables."""
        self.log("\n" + "="*60)
        self.log("GENERATING FINAL REPORT")
        self.log("="*60)

        from ccece.experiments.explainability.report_generator import ReportGenerator

        generator = ReportGenerator(
            results=self.results,
            output_dir=self.dirs['base'],
            tables_dir=self.dirs['tables'],
            channel_names=CHANNEL_NAMES,
            channel_categories=CHANNEL_CATEGORIES,
        )

        generator.generate_all()

        self.log(f"\nReport generated in: {self.dirs['base']}")

    def run_all(self):
        """Run the complete explainability analysis."""
        start_time = datetime.now()

        self.log("\n" + "="*70)
        self.log("EXPERIMENT 03: EXPLAINABILITY ANALYSIS")
        self.log("="*70)
        self.log(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Prepare data and train model
        self.prepare_data()
        self.train_model()

        # Run all components
        self.run_prototype_analysis()
        self.run_saliency_analysis()
        self.run_feature_importance()
        self.run_head_analysis()
        self.run_case_studies()

        # Generate report
        self.generate_report()

        end_time = datetime.now()
        duration = end_time - start_time

        self.log("\n" + "="*70)
        self.log("EXPLAINABILITY ANALYSIS COMPLETE")
        self.log("="*70)
        self.log(f"Duration: {duration}")
        self.log(f"Results saved to: {self.output_dir}")

        return self.results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CCECE Experiment 03: Explainability Analysis'
    )
    parser.add_argument(
        '--component',
        type=str,
        default=None,
        choices=['prototype', 'saliency', 'features', 'heads', 'cases'],
        help='Run only a specific component'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    args = parser.parse_args()

    # Set seeds for reproducibility
    set_all_seeds(RANDOM_SEED)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_BASE_DIR, timestamp)

    # Create configuration
    config = ExplainabilityConfig()

    # Create runner
    runner = ExplainabilityRunner(
        config=config,
        output_dir=output_dir,
        device=device,
        verbose=not args.quiet,
    )

    # Prepare data and train model (always needed)
    runner.prepare_data()
    runner.train_model()

    # Run analysis
    if args.component:
        runner.run_component(args.component)
    else:
        runner.run_all()

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
