"""
Main Runner for Ablation Study

Runs all 21 ablation variants with 5-fold cross-validation (105 total runs).
Uses IDENTICAL fold indices as EXPERIMENT 01 for fair comparison.
"""

import os
import sys
import json
import time
import random
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import SequenceScaler

from ccece.experiments.ablation_study.configs import (
    AblationConfig,
    get_all_ablation_variants,
    get_default_config,
    ClassificationType,
)
from ccece.experiments.ablation_study.variants.ablation_model import (
    AblationMultiHeadProtoNet,
    create_model_for_ablation,
)

# Try to import from sota_comparison for metrics
try:
    from ccece.experiments.sota_comparison.metrics import (
        ClassificationMetrics,
        compute_classification_metrics,
        aggregate_metrics,
    )
except ImportError:
    from sota_comparison.metrics import (
        ClassificationMetrics,
        compute_classification_metrics,
        aggregate_metrics,
    )


RANDOM_SEED = 42


class AblationTrainer:
    """Trainer for ablation study variants."""

    def __init__(
        self,
        model: AblationMultiHeadProtoNet,
        config: AblationConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.best_model_state = None

    def _setup_training(self, train_labels: np.ndarray):
        """Setup optimizer, scheduler, and loss function."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # Class weights for imbalanced data
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_labels: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the model."""
        self._setup_training(train_labels)
        patience_counter = 0

        epoch_iter = range(self.config.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch", leave=False)

        for epoch in epoch_iter:
            # Training
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss, val_acc = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Track best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            # Update progress bar
            if verbose and hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix({
                    'loss': f'{train_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'best': f'{self.best_val_accuracy:.4f}',
                })

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
        }

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch with configurable loss."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward with explanations
            logits, z, all_distances, all_similarities = self.model.forward_with_explanations(inputs)

            # Main classification loss (on averaged logits)
            loss = torch.tensor(0.0, device=self.device)

            if self.config.use_ce_loss:
                ce_loss = self.criterion(logits, labels)
                loss = loss + ce_loss

            # Per-head classification loss (if enabled)
            if self.config.per_head_ce_weight > 0:
                per_head_ce_loss = torch.tensor(0.0, device=self.device)
                for h_idx in range(self.model.n_heads):
                    head_logits = all_similarities[h_idx]
                    head_ce = self.criterion(head_logits, labels)
                    per_head_ce_loss = per_head_ce_loss + head_ce
                per_head_ce_loss = per_head_ce_loss / self.model.n_heads
                loss = loss + self.config.per_head_ce_weight * per_head_ce_loss

            # Prototype losses (only if using prototypes)
            if self.model.use_prototypes:
                cluster_loss, separation_loss = self.model.compute_prototype_loss(z, labels)

                if self.config.use_cluster_loss:
                    loss = loss + self.config.cluster_loss_weight * cluster_loss

                if self.config.use_separation_loss:
                    loss = loss + self.config.separation_loss_weight * separation_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm
            )

            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        return total_loss / total_samples

    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def evaluate(self, dataloader: DataLoader) -> ClassificationMetrics:
        """Evaluate the model and compute metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        probabilities = np.array(all_probs)

        return compute_classification_metrics(labels, predictions, probabilities)


def prepare_data_for_fold(
    items: List[Dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """
    Prepare train and validation data loaders for a fold.
    """
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]

    # Fit scaler on training data
    scaler = SequenceScaler().fit(train_items)

    # Process data
    def process_items(items_list, scaler, seq_len):
        X_list = []
        y_list = []
        for item in items_list:
            data = scaler.transform(item['data'].copy())
            # Pad or truncate
            if len(data) >= seq_len:
                data = data[:seq_len]
            else:
                padding = np.zeros((seq_len - len(data), data.shape[1]), dtype=np.float32)
                data = np.vstack([data, padding])
            X_list.append(data)
            y_list.append(item['label'])
        return np.array(X_list, dtype=np.float32), np.array(y_list)

    X_train, y_train = process_items(train_items, scaler, seq_len)
    X_val, y_val = process_items(val_items, scaler, seq_len)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, y_train


def store_training_embeddings(
    model: AblationMultiHeadProtoNet,
    train_loader: DataLoader,
    device: torch.device,
):
    """Store training embeddings for alignment computation."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            z = model.encode(inputs)
            all_embeddings.append(z.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    model.store_training_embeddings(embeddings, labels)


def run_single_variant_fold(
    config: AblationConfig,
    items: List[Dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seq_len: int,
    input_dim: int,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single ablation variant on a single fold.

    Returns metrics and alignment info.
    """
    # Prepare data
    train_loader, val_loader, train_labels = prepare_data_for_fold(
        items, train_idx, val_idx, seq_len, config.batch_size
    )

    # Create model
    model = create_model_for_ablation(
        input_dim=input_dim,
        num_classes=2,
        seq_len=seq_len,
        config=config,
    )

    # Train
    start_time = time.time()
    trainer = AblationTrainer(model, config, device)
    train_result = trainer.train(train_loader, val_loader, train_labels, verbose=verbose)
    training_time = time.time() - start_time

    # Evaluate
    metrics = trainer.evaluate(val_loader)

    # Compute alignment if using prototypes
    alignment_metrics = {}
    if model.use_prototypes:
        store_training_embeddings(model, train_loader, device)
        per_head = model.compute_alignment_per_head()
        alignments = [per_head[i]['alignment'] for i in range(model.n_heads)]

        alignment_metrics = {
            'overall_alignment': np.mean(alignments),
            'std_alignment': np.std(alignments),
            'min_alignment': np.min(alignments),
            'max_alignment': np.max(alignments),
            'per_head_alignments': alignments,
        }

    # Compute parameters
    num_params = sum(p.numel() for p in model.parameters())

    return {
        'metrics': metrics,
        'alignment': alignment_metrics,
        'training_time': training_time,
        'best_epoch': train_result['best_epoch'],
        'num_params': num_params,
    }


def run_ablation_study(
    output_dir: Optional[str] = None,
    n_folds: int = 5,
    ablations_to_run: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full ablation study.

    Args:
        output_dir: Output directory (auto-generated if None)
        n_folds: Number of cross-validation folds
        ablations_to_run: List of ablations to run (None = all)
        verbose: Print progress

    Returns:
        Dict with all results
    """
    set_all_seeds(RANDOM_SEED)

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/ablation_study/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 70)
        print("ABLATION STUDY EXPERIMENT")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Output: {output_dir}")
        print(f"Folds: {n_folds}")
        print()

    # Load and preprocess data
    if verbose:
        print("Loading data...")
    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)

    X, y, patient_ids = extract_arrays(items)
    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]

    if verbose:
        print(f"Data: {len(items)} samples, seq_len={seq_len}, input_dim={input_dim}")
        print(f"Class distribution: HC={sum(y==0)}, MG={sum(y==1)}")
        print()

    # Get all ablation variants
    all_variants = get_all_ablation_variants()

    # Filter ablations if specified
    if ablations_to_run is not None:
        all_variants = {k: v for k, v in all_variants.items() if k in ablations_to_run}

    # Count total variants
    total_variants = sum(len(v) for v in all_variants.values())
    total_runs = total_variants * n_folds

    if verbose:
        print(f"Ablations: {list(all_variants.keys())}")
        print(f"Total variants: {total_variants}")
        print(f"Total training runs: {total_runs}")
        print()

    # Setup cross-validation with SAME seed as EXPERIMENT 01
    if n_folds == 1:
        # Use GroupShuffleSplit for single fold (for quick testing)
        from sklearn.model_selection import GroupShuffleSplit
        cv = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    else:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    # Store fold indices for reproducibility
    fold_indices = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, patient_ids)):
        fold_indices.append({
            'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(),
        })

    # Results storage
    all_results = {}
    run_count = 0

    # Run each ablation
    for ablation_name, variants in all_variants.items():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"ABLATION: {ablation_name.upper()}")
            print(f"{'=' * 60}")

        ablation_results = {}

        for variant in variants:
            variant_name = variant.variant_name
            marker = " (DEFAULT)" if variant.is_default else ""

            if verbose:
                print(f"\n--- Variant: {variant_name}{marker} ---")

            variant_fold_results = []

            for fold_idx in range(n_folds):
                run_count += 1
                train_idx = np.array(fold_indices[fold_idx]['train_idx'])
                val_idx = np.array(fold_indices[fold_idx]['val_idx'])

                if verbose:
                    print(f"  Fold {fold_idx + 1}/{n_folds} (run {run_count}/{total_runs})...", end=" ", flush=True)

                try:
                    fold_result = run_single_variant_fold(
                        config=variant,
                        items=items,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        seq_len=seq_len,
                        input_dim=input_dim,
                        device=device,
                        verbose=False,
                    )

                    variant_fold_results.append({
                        'fold': fold_idx + 1,
                        'accuracy': fold_result['metrics'].accuracy,
                        'balanced_accuracy': fold_result['metrics'].balanced_accuracy,
                        'sensitivity': fold_result['metrics'].sensitivity,
                        'specificity': fold_result['metrics'].specificity,
                        'f1_score': fold_result['metrics'].f1_score,
                        'auc_roc': fold_result['metrics'].auc_roc,
                        'alignment': fold_result['alignment'],
                        'training_time': fold_result['training_time'],
                        'best_epoch': fold_result['best_epoch'],
                        'num_params': fold_result['num_params'],
                    })

                    if verbose:
                        print(f"acc={fold_result['metrics'].accuracy:.3f}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    variant_fold_results.append({
                        'fold': fold_idx + 1,
                        'error': str(e),
                    })

                # Clear GPU memory
                torch.cuda.empty_cache()

            # Aggregate variant results
            valid_results = [r for r in variant_fold_results if 'error' not in r]

            if valid_results:
                aggregated = {
                    'accuracy': {
                        'mean': np.mean([r['accuracy'] for r in valid_results]),
                        'std': np.std([r['accuracy'] for r in valid_results]),
                        'values': [r['accuracy'] for r in valid_results],
                    },
                    'balanced_accuracy': {
                        'mean': np.mean([r['balanced_accuracy'] for r in valid_results]),
                        'std': np.std([r['balanced_accuracy'] for r in valid_results]),
                        'values': [r['balanced_accuracy'] for r in valid_results],
                    },
                    'sensitivity': {
                        'mean': np.mean([r['sensitivity'] for r in valid_results]),
                        'std': np.std([r['sensitivity'] for r in valid_results]),
                        'values': [r['sensitivity'] for r in valid_results],
                    },
                    'specificity': {
                        'mean': np.mean([r['specificity'] for r in valid_results]),
                        'std': np.std([r['specificity'] for r in valid_results]),
                        'values': [r['specificity'] for r in valid_results],
                    },
                    'f1_score': {
                        'mean': np.mean([r['f1_score'] for r in valid_results]),
                        'std': np.std([r['f1_score'] for r in valid_results]),
                        'values': [r['f1_score'] for r in valid_results],
                    },
                    'auc_roc': {
                        'mean': np.mean([r['auc_roc'] for r in valid_results]),
                        'std': np.std([r['auc_roc'] for r in valid_results]),
                        'values': [r['auc_roc'] for r in valid_results],
                    },
                    'training_time': {
                        'mean': np.mean([r['training_time'] for r in valid_results]),
                        'std': np.std([r['training_time'] for r in valid_results]),
                    },
                    'num_params': valid_results[0]['num_params'],
                }

                # Add alignment if available
                alignments = [r['alignment'].get('overall_alignment', 0) for r in valid_results if r['alignment']]
                if alignments:
                    aggregated['alignment'] = {
                        'mean': np.mean(alignments),
                        'std': np.std(alignments),
                        'values': alignments,
                    }

                if verbose:
                    print(f"  Summary: {aggregated['accuracy']['mean']*100:.1f} ± {aggregated['accuracy']['std']*100:.1f}%")

            else:
                aggregated = {'error': 'All folds failed'}

            ablation_results[variant_name] = {
                'config': variant.to_dict(),
                'is_default': variant.is_default,
                'fold_results': variant_fold_results,
                'aggregated': aggregated,
            }

        all_results[ablation_name] = ablation_results

        # Save intermediate results after each ablation
        _save_intermediate_results(all_results, output_dir, ablation_name)

    # Generate final output
    if verbose:
        print("\n" + "=" * 70)
        print("GENERATING FINAL OUTPUT")
        print("=" * 70)

    # Import analysis module and generate tables
    try:
        from ccece.experiments.ablation_study.analysis import generate_all_ablation_tables
        generate_all_ablation_tables(all_results, output_dir, verbose=verbose)
    except Exception as e:
        print(f"Warning: Could not generate tables: {e}")

    # Save full results
    full_results = {
        'config': {
            'random_seed': RANDOM_SEED,
            'n_folds': n_folds,
            'seq_len': seq_len,
            'input_dim': input_dim,
            'n_samples': len(items),
        },
        'fold_indices': fold_indices,
        'results': _convert_to_serializable(all_results),
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    # Print summary
    if verbose:
        _print_summary(all_results)
        print(f"\nResults saved to: {output_dir}")

    return full_results


def _save_intermediate_results(results: Dict, output_dir: str, ablation_name: str):
    """Save intermediate results after each ablation."""
    ablation_dir = os.path.join(output_dir, 'per_ablation', f'ablation_{ablation_name}')
    os.makedirs(ablation_dir, exist_ok=True)

    results_path = os.path.join(ablation_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(_convert_to_serializable(results.get(ablation_name, {})), f, indent=2)


def _convert_to_serializable(obj):
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
        return {str(k): _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    return obj


def _print_summary(all_results: Dict):
    """Print summary of ablation study results."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)

    for ablation_name, variants in all_results.items():
        print(f"\n{ablation_name.upper()}:")
        print("-" * 50)

        # Find default for comparison
        default_acc = None
        for variant_name, data in variants.items():
            if data.get('is_default'):
                default_acc = data['aggregated']['accuracy']['mean']
                break

        for variant_name, data in variants.items():
            if 'error' in data.get('aggregated', {}):
                print(f"  {variant_name:20s}: ERROR")
                continue

            acc = data['aggregated']['accuracy']
            marker = " (DEFAULT)" if data.get('is_default') else ""

            # Compute delta vs default
            if default_acc is not None and not data.get('is_default'):
                delta = (acc['mean'] - default_acc) * 100
                delta_str = f" ({delta:+.1f}%)"
            else:
                delta_str = ""

            print(f"  {variant_name:20s}: {acc['mean']*100:.1f} ± {acc['std']*100:.1f}%{delta_str}{marker}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ablation Study for MultiHeadProtoNet')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--ablations', type=str, nargs='+', default=None,
                        choices=['n_heads', 'classification', 'loss', 'fusion', 'encoder'],
                        help='Ablations to run (default: all)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')

    args = parser.parse_args()

    run_ablation_study(
        output_dir=args.output,
        n_folds=args.folds,
        ablations_to_run=args.ablations,
        verbose=not args.quiet,
    )
