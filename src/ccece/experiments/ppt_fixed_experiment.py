"""
CCECE Paper: PPT Fixed Experiment

Tests the PPT model with relative temporal segmentation to fix the padding artifact.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays, add_engineered_features, subsample_sequence
from ccece.run_experiment import set_all_seeds
from ccece.trainer import EvaluationMetrics
from ccece.models.ppt_fixed import PPTFixed


RANDOM_SEED = 42


@dataclass
class PPTFixedConfig:
    latent_dim: int = 64
    n_prototypes_per_class: int = 5
    n_segments: int = 8
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2
    trajectory_type: str = 'linear'
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0
    cluster_loss_weight: float = 0.5
    separation_loss_weight: float = 0.1
    diversity_loss_weight: float = 0.05
    n_folds: int = 5


class VariableLengthDataset(Dataset):
    """Dataset that returns actual sequence lengths."""

    def __init__(self, items: List[Dict], max_len: int):
        self.items = items
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        data = item['data']
        label = item['label']
        actual_len = len(data)

        # Pad to max_len
        if actual_len < self.max_len:
            padding = np.zeros((self.max_len - actual_len, data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        else:
            data = data[:self.max_len]
            actual_len = self.max_len

        return (
            torch.from_numpy(data).float(),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(actual_len, dtype=torch.long),
        )


def create_data_loaders_with_lengths(
    train_items: List[Dict],
    val_items: List[Dict],
    max_len: int,
    batch_size: int,
):
    """Create data loaders that include sequence lengths."""
    # Normalize using training data
    all_train_data = np.vstack([item['data'] for item in train_items])
    mean = all_train_data.mean(axis=0)
    std = all_train_data.std(axis=0) + 1e-8

    for item in train_items:
        item['data'] = (item['data'] - mean) / std
    for item in val_items:
        item['data'] = (item['data'] - mean) / std

    train_dataset = VariableLengthDataset(train_items, max_len)
    val_dataset = VariableLengthDataset(val_items, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class PPTFixedTrainer:
    def __init__(self, model: PPTFixed, config: PPTFixedConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.best_val_accuracy = 0.0
        self.best_model_state = None

    def train(self, train_loader, val_loader, train_labels, verbose=True):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        patience_counter = 0
        epoch_iter = range(self.config.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            # Train
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels, lengths in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                optimizer.zero_grad()

                logits, z_segments, distances, per_seg_sims, traj_sims, segment_mask = \
                    self.model.forward_with_explanations(inputs, lengths)

                ce_loss = criterion(logits, labels)
                cluster_loss, separation_loss, diversity_loss = \
                    self.model.compute_prototype_loss(z_segments, labels, distances, segment_mask)

                loss = (ce_loss +
                       self.config.cluster_loss_weight * cluster_loss +
                       self.config.separation_loss_weight * separation_loss +
                       self.config.diversity_loss_weight * diversity_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_loss = total_loss / total
            train_acc = correct / total

            # Validate
            val_loss, val_acc = self._validate(val_loader, criterion)
            scheduler.step(val_loss)

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose:
                v_norm = self.model.prototype_layer.compute_velocity_norms().mean().item()
                epoch_iter.set_postfix({
                    'tr_loss': f'{train_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'best': f'{self.best_val_accuracy:.4f}',
                    'v_norm': f'{v_norm:.4f}',
                })

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return {'best_val_accuracy': self.best_val_accuracy}

    def _validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels, lengths in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                logits = self.model(inputs, lengths)
                loss = criterion(logits, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def evaluate(self, dataloader) -> EvaluationMetrics:
        from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels, lengths in dataloader:
                inputs = inputs.to(self.device)
                lengths = lengths.to(self.device)

                outputs = self.model(inputs, lengths)
                probs = torch.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        probabilities = np.array(all_probs)

        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        return EvaluationMetrics(
            accuracy=accuracy_score(labels, predictions),
            sensitivity=tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            precision=precision_score(labels, predictions, zero_division=0),
            f1=f1_score(labels, predictions, zero_division=0),
            auc_roc=roc_auc_score(labels, probabilities),
            confusion_matrix=cm,
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
        )


def run_temporal_pattern_test(model, val_loader, device):
    """
    Test whether late segments are more discriminative than early segments.

    This is the key test that failed before due to padding.
    With relative segmentation, late segments now contain actual late data.
    """
    model.eval()
    n_segments = model.n_segments
    prototype_classes = model.get_prototype_classes()

    early_seg_diffs = []
    late_seg_diffs = []

    with torch.no_grad():
        for inputs, labels, lengths in val_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)

            per_seg_sims, segment_mask = model.get_per_segment_similarities(inputs, lengths)

            for i in range(inputs.size(0)):
                label = labels[i].item()
                mg_mask = (prototype_classes == 1)
                hc_mask = (prototype_classes == 0)

                # Per-segment discrimination
                seg_mg_sims = per_seg_sims[i, :, mg_mask].mean(dim=1).cpu().numpy()
                seg_hc_sims = per_seg_sims[i, :, hc_mask].mean(dim=1).cpu().numpy()

                if label == 1:  # MG
                    seg_diff = seg_mg_sims - seg_hc_sims
                else:  # HC
                    seg_diff = seg_hc_sims - seg_mg_sims

                half = n_segments // 2
                early_seg_diffs.append(seg_diff[:half].mean())
                late_seg_diffs.append(seg_diff[half:].mean())

    early_discrimination = np.mean(early_seg_diffs)
    late_discrimination = np.mean(late_seg_diffs)

    return {
        'early_discrimination': early_discrimination,
        'late_discrimination': late_discrimination,
        'late_minus_early': late_discrimination - early_discrimination,
        'temporal_pattern_pass': late_discrimination > early_discrimination,
    }


def run_trajectory_ablation(model, val_loader, device):
    """Compare trajectory vs static prototypes."""
    # Trajectory accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, lengths in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(inputs, lengths)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    trajectory_acc = correct / total

    # Static accuracy (zero velocities)
    original_velocities = model.prototype_layer.prototype_velocities.data.clone()
    model.prototype_layer.prototype_velocities.data.zero_()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, lengths in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(inputs, lengths)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    static_acc = correct / total
    model.prototype_layer.prototype_velocities.data = original_velocities

    return {
        'trajectory_accuracy': trajectory_acc,
        'static_accuracy': static_acc,
        'trajectory_improvement': trajectory_acc - static_acc,
        'trajectory_helps': trajectory_acc > static_acc + 0.005,
    }


def run_experiment(config: PPTFixedConfig, output_dir: str, verbose: bool = True, n_folds: int = 3):
    set_all_seeds(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print(f"Running {n_folds}-fold cross-validation with RELATIVE temporal segmentation")

    # Load data
    if verbose:
        print("Loading data...")
    items = load_binary_dataset(verbose=False)

    # Preprocess: add features and subsample
    for item in items:
        item['data'] = add_engineered_features(item['data'])
        item['data'] = subsample_sequence(item['data'], factor=10)

    X, y, patient_ids = extract_arrays(items)
    max_len = int(np.percentile([len(item['data']) for item in items], 90))
    input_dim = items[0]['data'].shape[1]

    if verbose:
        print(f"Data: {len(items)} samples, max_len={max_len}, input_dim={input_dim}")

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    best_model = None
    best_accuracy = 0.0
    best_temporal_test = None
    best_ablation = None

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, patient_ids)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{n_folds}")
            print('='*50)

        train_items = [items[i].copy() for i in train_idx]
        val_items = [items[i].copy() for i in val_idx]
        train_labels = np.array([item['label'] for item in train_items])

        train_loader, val_loader = create_data_loaders_with_lengths(
            train_items, val_items, max_len, config.batch_size
        )

        model = PPTFixed(
            input_dim=input_dim,
            num_classes=2,
            seq_len=max_len,
            latent_dim=config.latent_dim,
            n_prototypes_per_class=config.n_prototypes_per_class,
            n_segments=config.n_segments,
            encoder_hidden=config.encoder_hidden,
            encoder_layers=config.encoder_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            trajectory_type=config.trajectory_type,
        )

        if verbose:
            print(f"Model parameters: {model.count_parameters():,}")

        trainer = PPTFixedTrainer(model, config, device)
        trainer.train(train_loader, val_loader, train_labels, verbose=verbose)

        metrics = trainer.evaluate(val_loader)
        if verbose:
            print(f"\nFold {fold + 1} Results: {metrics}")

        # Temporal pattern test
        temporal_test = run_temporal_pattern_test(model, val_loader, device)
        if verbose:
            print(f"\nTemporal Pattern Test:")
            print(f"  Early discrimination: {temporal_test['early_discrimination']:.4f}")
            print(f"  Late discrimination: {temporal_test['late_discrimination']:.4f}")
            print(f"  Late - Early: {temporal_test['late_minus_early']:.4f}")
            print(f"  PASS (late > early): {temporal_test['temporal_pattern_pass']}")

        # Ablation
        ablation = run_trajectory_ablation(model, val_loader, device)
        if verbose:
            print(f"\nTrajectory Ablation:")
            print(f"  Trajectory: {ablation['trajectory_accuracy']:.4f}")
            print(f"  Static: {ablation['static_accuracy']:.4f}")
            print(f"  Improvement: {ablation['trajectory_improvement']:.4f}")
            print(f"  Trajectory helps: {ablation['trajectory_helps']}")

        fold_results.append({
            'fold': fold + 1,
            'accuracy': metrics.accuracy,
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
            'f1': metrics.f1,
            'auc_roc': metrics.auc_roc,
            **temporal_test,
            **ablation,
        })

        if metrics.accuracy > best_accuracy:
            best_accuracy = metrics.accuracy
            best_model = model
            best_temporal_test = temporal_test
            best_ablation = ablation

    # Summary
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    mean_auc = np.mean([r['auc_roc'] for r in fold_results])

    # Temporal pattern across folds
    temporal_pass_count = sum(1 for r in fold_results if r['temporal_pattern_pass'])
    trajectory_helps_count = sum(1 for r in fold_results if r['trajectory_helps'])

    summary = {
        'config': asdict(config),
        'n_folds': n_folds,
        'fix_applied': 'relative_temporal_segmentation',
        'results': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_auc': mean_auc,
        },
        'temporal_pattern': {
            'folds_passing': temporal_pass_count,
            'total_folds': n_folds,
            'mean_late_minus_early': np.mean([r['late_minus_early'] for r in fold_results]),
        },
        'trajectory_ablation': {
            'folds_trajectory_helps': trajectory_helps_count,
            'mean_improvement': np.mean([r['trajectory_improvement'] for r in fold_results]),
        },
        'fold_results': fold_results,
        'timestamp': datetime.now().isoformat(),
    }

    if verbose:
        print("\n" + "="*60)
        print("PPT FIXED EXPERIMENT SUMMARY")
        print("="*60)
        print(f"\nPerformance:")
        print(f"  Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"  Mean AUC-ROC: {mean_auc:.4f}")

        print(f"\nTemporal Pattern Test (KEY FIX VALIDATION):")
        print(f"  Folds where late > early: {temporal_pass_count}/{n_folds}")
        print(f"  Mean (late - early): {summary['temporal_pattern']['mean_late_minus_early']:.4f}")
        if temporal_pass_count == n_folds:
            print(f"  STATUS: FIXED - Fatigue pattern now detected!")
        elif temporal_pass_count > 0:
            print(f"  STATUS: PARTIALLY FIXED")
        else:
            print(f"  STATUS: NOT FIXED")

        print(f"\nTrajectory Ablation:")
        print(f"  Folds where trajectory > static: {trajectory_helps_count}/{n_folds}")
        print(f"  Mean improvement: {summary['trajectory_ablation']['mean_improvement']:.4f}")

    # Save
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    if verbose:
        print(f"\nResults saved to: {output_dir}")

    return summary


if __name__ == '__main__':
    config = PPTFixedConfig()
    output_dir = f"results/ccece/ppt_fixed/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_experiment(config, output_dir, verbose=True, n_folds=3)
