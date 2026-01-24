"""
CCECE Paper: TAProtoNet Experiment

Rapid prototype experiment for Temporally-Attentive Prototype Network.

PRIMARY SUCCESS CRITERION (IMMUTABLE): Accuracy > 71.70% (TempProtoNet baseline)

This is the third approach after PPT (67.67%) and HMSProtoNet (68.65%) failed.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import (
    EvaluationMetrics, create_data_loaders
)
from ccece.models.ta_proto_net import TAProtoNet


RANDOM_SEED = 42
BASELINE_ACCURACY = 0.7170


@dataclass
class TAProtoNetConfig:
    """Configuration for TAProtoNet experiment."""
    # Model hyperparameters
    latent_dim: int = 64
    n_prototypes_per_class: int = 5  # Same as baseline
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2
    n_segments: int = 8  # Temporal segments for attention

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0

    # Prototype loss weights
    cluster_loss_weight: float = 0.5
    separation_loss_weight: float = 0.1
    temporal_loss_weight: float = 0.0  # Start without temporal constraint

    # Cross-validation
    n_folds: int = 3


class TAProtoNetTrainer:
    """Trainer for TAProtoNet."""

    def __init__(
        self,
        model: TAProtoNet,
        config: TAProtoNetConfig,
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
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5,
        )

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
        self._setup_training(train_labels)
        patience_counter = 0

        epoch_iter = range(self.config.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss, val_acc = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            self.scheduler.step(val_loss)

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose:
                epoch_iter.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'best': f'{self.best_val_accuracy:.4f}',
                })

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
        }

    def _train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model.forward_with_explanations(inputs)
            logits = outputs['logits']
            z = outputs['z']
            distances = outputs['distances']
            attention = outputs['attention_weights']

            # Classification loss
            ce_loss = self.criterion(logits, labels)

            # Prototype losses
            cluster_loss, separation_loss = self.model.compute_prototype_loss(
                z, labels, distances
            )

            # Optional temporal loss
            if self.config.temporal_loss_weight > 0:
                temporal_loss = self.model.compute_temporal_loss(attention, labels)
            else:
                temporal_loss = torch.tensor(0.0, device=self.device)

            loss = (
                ce_loss +
                self.config.cluster_loss_weight * cluster_loss +
                self.config.separation_loss_weight * separation_loss +
                self.config.temporal_loss_weight * temporal_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

        return total_loss / total

    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
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

    def evaluate(self, dataloader: DataLoader) -> EvaluationMetrics:
        from sklearn.metrics import (
            accuracy_score, precision_score, f1_score,
            roc_auc_score, confusion_matrix
        )

        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
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


def run_rapid_prototype(config: TAProtoNetConfig, output_dir: str, verbose: bool = True):
    """Run rapid prototype experiment."""
    set_all_seeds(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print(f"\n{'='*60}")
        print("PRIMARY CRITERION: Accuracy > 71.70% (TempProtoNet baseline)")
        print("Previous failures: PPT (67.67%), HMSProtoNet (68.65%)")
        print(f"{'='*60}\n")

    if verbose:
        print("Loading data...")
    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)

    X, y, patient_ids = extract_arrays(items)
    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]
    num_classes = 2

    if verbose:
        print(f"Data: {len(items)} samples, seq_len={seq_len}, input_dim={input_dim}")

    cv = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, patient_ids)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{config.n_folds}")
            print('='*50)

        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        train_labels = np.array([item['label'] for item in train_items])

        train_loader, val_loader, scaler = create_data_loaders(
            train_items, val_items, seq_len, config.batch_size
        )

        model = TAProtoNet(
            input_dim=input_dim,
            num_classes=num_classes,
            seq_len=seq_len,
            latent_dim=config.latent_dim,
            n_prototypes_per_class=config.n_prototypes_per_class,
            encoder_hidden=config.encoder_hidden,
            encoder_layers=config.encoder_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            n_segments=config.n_segments,
        )

        if verbose:
            print(f"Model parameters: {model.count_parameters():,}")

        trainer = TAProtoNetTrainer(model, config, device)
        train_result = trainer.train(train_loader, val_loader, train_labels, verbose=verbose)

        metrics = trainer.evaluate(val_loader)

        if verbose:
            print(f"\nFold {fold + 1} Results:")
            print(f"  Accuracy: {metrics.accuracy:.4f}")
            print(f"  AUC-ROC: {metrics.auc_roc:.4f}")
            print(f"  F1: {metrics.f1:.4f}")

        # Analyze attention patterns
        model.eval()
        all_attention = []
        all_labels_list = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                attn = model.get_attention_weights(inputs)
                all_attention.append(attn.cpu())
                all_labels_list.extend(labels.numpy())

        all_attention = torch.cat(all_attention, dim=0)
        all_labels_arr = np.array(all_labels_list)

        # Compute mean attention per segment for MG vs HC
        mg_attention = all_attention[all_labels_arr == 1].mean(dim=0).numpy()
        hc_attention = all_attention[all_labels_arr == 0].mean(dim=0).numpy()

        if verbose:
            print(f"  MG attention (first half): {mg_attention[:config.n_segments//2].mean():.3f}")
            print(f"  MG attention (second half): {mg_attention[config.n_segments//2:].mean():.3f}")
            print(f"  HC attention (first half): {hc_attention[:config.n_segments//2].mean():.3f}")
            print(f"  HC attention (second half): {hc_attention[config.n_segments//2:].mean():.3f}")

        fold_result = {
            'fold': fold + 1,
            'accuracy': metrics.accuracy,
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
            'f1': metrics.f1,
            'auc_roc': metrics.auc_roc,
            'best_epoch': train_result['best_epoch'],
            'mg_attention_first_half': float(mg_attention[:config.n_segments//2].mean()),
            'mg_attention_second_half': float(mg_attention[config.n_segments//2:].mean()),
        }
        fold_results.append(fold_result)

    # Aggregate
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    mean_auc = np.mean([r['auc_roc'] for r in fold_results])
    mean_f1 = np.mean([r['f1'] for r in fold_results])

    primary_criterion_passed = mean_accuracy > BASELINE_ACCURACY

    summary = {
        'config': asdict(config),
        'results': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_auc': mean_auc,
            'mean_f1': mean_f1,
        },
        'fold_results': fold_results,
        'primary_criterion': {
            'baseline': BASELINE_ACCURACY,
            'achieved': mean_accuracy,
            'passed': primary_criterion_passed,
        },
        'timestamp': datetime.now().isoformat(),
    }

    print("\n" + "="*60)
    print("PRIMARY CRITERION EVALUATION")
    print("="*60)
    print(f"\nBaseline (TempProtoNet): {BASELINE_ACCURACY:.4f} (71.70%)")
    print(f"TAProtoNet achieved:     {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
    print(f"Standard deviation:      {std_accuracy:.4f}")
    print()

    if primary_criterion_passed:
        print(">>> PRIMARY CRITERION: PASSED <<<")
        print("TAProtoNet outperforms baseline. Proceeding to full evaluation.")
    else:
        print(">>> PRIMARY CRITERION: FAILED <<<")
        print("TAProtoNet does NOT outperform baseline.")
        print("Per protocol: STOP. Do not investigate. PIVOT to next approach.")

    print("="*60)

    # Save
    results_path = os.path.join(output_dir, 'rapid_prototype_results.json')

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


if __name__ == '__main__':
    config = TAProtoNetConfig(
        latent_dim=64,
        n_prototypes_per_class=5,
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
        n_segments=8,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=15,
        cluster_loss_weight=0.5,
        separation_loss_weight=0.1,
        temporal_loss_weight=0.0,
        n_folds=3,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/ccece/ta_proto_net/rapid_prototype_{timestamp}"

    summary = run_rapid_prototype(config, output_dir, verbose=True)
