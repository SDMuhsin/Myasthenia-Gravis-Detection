"""
Ablation Study Runner for MultiHeadTrajectoryProtoNet

Runs all 70 training configurations (14 variants x 5 folds) with:
- Incremental saving after each variant
- Support for static prototypes (frozen velocities)
- Support for different segment weighting strategies
- Reproducible cross-validation with random_state=42
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import TrainingConfig, EvaluationMetrics, create_data_loaders
from ccece.models.multi_head_trajectory_proto_net import (
    MultiHeadTrajectoryProtoNet,
    TrajectoryPrototypeHead,
)

from .configs import (
    AblationConfig,
    SegmentWeightingStrategy,
    get_all_ablation_configs,
)


RANDOM_SEED = 42


class LearnedAttentionWeighter(nn.Module):
    """Learned attention weights for segment weighting (Ablation 4.3)."""

    def __init__(self, n_segments: int, latent_dim: int):
        super().__init__()
        self.n_segments = n_segments
        # Query vector for attention
        self.query = nn.Parameter(torch.randn(latent_dim) * 0.1)

    def forward(self, z_segments: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights from segment encodings.

        Args:
            z_segments: (batch, n_segments, latent_dim)

        Returns:
            weights: (batch, n_segments) normalized attention weights
        """
        # Compute attention scores
        scores = torch.einsum('bsd,d->bs', z_segments, self.query)
        weights = F.softmax(scores, dim=1)
        return weights


class AblationMHTPN(MultiHeadTrajectoryProtoNet):
    """
    Extended MHTPN with ablation-specific modifications.

    Supports:
    - Static prototypes (frozen velocities)
    - Different segment weighting strategies
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        config: AblationConfig,
    ):
        # Initialize base model
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
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

        self.ablation_config = config
        self.segment_weighting_strategy = config.segment_weighting

        # Handle static prototypes (ablation 2)
        if not config.use_trajectory_prototypes:
            self._freeze_velocities()

        # Handle learned attention weighting (ablation 4.3)
        if config.segment_weighting == SegmentWeightingStrategy.LEARNED_ATTENTION:
            self.attention_weighter = LearnedAttentionWeighter(
                n_segments=config.n_segments,
                latent_dim=config.latent_dim,
            )
        else:
            self.attention_weighter = None

    def _freeze_velocities(self):
        """Freeze all prototype velocities to zero for static prototypes."""
        for head in self.heads:
            # Set velocities to zero
            head.prototype_velocities.data.zero_()
            # Make non-trainable
            head.prototype_velocities.requires_grad = False

    def compute_segment_weights(
        self,
        lengths: torch.Tensor,
        z_segments: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute segment weights based on the configured strategy.

        Args:
            lengths: (batch_size,) actual sequence lengths
            z_segments: (batch_size, n_segments, latent_dim) - needed for learned attention

        Returns:
            segment_weights: (batch_size, n_segments)
        """
        batch_size = lengths.size(0)
        device = lengths.device

        if self.segment_weighting_strategy == SegmentWeightingStrategy.UNIFORM:
            # All segments equal weight
            weights = torch.ones(batch_size, self.n_segments, device=device)
            weights = weights / self.n_segments

        elif self.segment_weighting_strategy == SegmentWeightingStrategy.PADDING_AWARE:
            # Weight by fraction of real data (default behavior)
            weights = super().compute_segment_weights(lengths)

        elif self.segment_weighting_strategy == SegmentWeightingStrategy.LEARNED_ATTENTION:
            # Use learned attention weights
            if z_segments is None:
                raise ValueError("z_segments required for learned attention weighting")
            weights = self.attention_weighter(z_segments)

        return weights

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with ablation-specific segment weighting."""
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        z_segments = self.encode_segments(x)

        # Get segment weights using the configured strategy
        segment_weights = self.compute_segment_weights(lengths, z_segments)

        # Collect logits from all heads
        all_logits = []
        for head in self.heads:
            _, _, _, head_logits = head(z_segments, segment_weights, self.t_default)
            all_logits.append(head_logits)

        # Average logits across heads
        logits = torch.stack(all_logits, dim=0).mean(dim=0)
        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass with explanations, using ablation-specific segment weighting."""
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        z_segments = self.encode_segments(x)
        segment_weights = self.compute_segment_weights(lengths, z_segments)

        all_logits = []
        all_per_seg_sims = []
        all_traj_sims = []

        for head in self.heads:
            _, per_seg_sims, traj_sims, head_logits = head(
                z_segments, segment_weights, self.t_default
            )
            all_logits.append(head_logits)
            all_per_seg_sims.append(per_seg_sims)
            all_traj_sims.append(traj_sims)

        logits = torch.stack(all_logits, dim=0).mean(dim=0)
        return logits, z_segments, segment_weights, all_per_seg_sims, all_traj_sims

    def get_velocity_norms(self) -> Dict[str, float]:
        """Get mean and std of velocity norms across all heads."""
        all_norms = []
        for head in self.heads:
            norms = head.get_velocity_norms()
            all_norms.extend(norms.detach().cpu().numpy().tolist())
        return {
            'mean_velocity_norm': float(np.mean(all_norms)),
            'std_velocity_norm': float(np.std(all_norms)),
            'all_velocity_norms': all_norms,
        }

    def compute_temporal_discrimination(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute early vs late segment discrimination.
        Override to handle learned attention weighting.
        """
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        self.eval()
        with torch.no_grad():
            z_segments = self.encode_segments(x)
            segment_weights = self.compute_segment_weights(lengths, z_segments)

            # Average per-segment similarities across heads
            all_per_seg_sims = []
            for head in self.heads:
                _, per_seg_sims, _, _ = head(z_segments, segment_weights, self.t_default)
                all_per_seg_sims.append(per_seg_sims)

            # (batch, n_segments, n_classes)
            avg_sims = torch.stack(all_per_seg_sims, dim=0).mean(dim=0)

            # Split by label
            hc_mask = (labels == 0)
            mg_mask = (labels == 1)

            # Get similarities to correct class prototype per segment
            hc_sims = avg_sims[hc_mask, :, 0].mean(dim=0)  # (n_segments,)
            mg_sims = avg_sims[mg_mask, :, 1].mean(dim=0)  # (n_segments,)

            # Discrimination = MG-to-MG similarity - HC-to-HC similarity at each segment
            discrimination = mg_sims - hc_sims

            # Split into early and late
            mid = self.n_segments // 2
            early_discrim = discrimination[:mid].mean().item()
            late_discrim = discrimination[mid:].mean().item()

            return {
                'early_discrimination': early_discrim,
                'late_discrimination': late_discrim,
                'late_minus_early': late_discrim - early_discrim,
                'temporal_pattern_pass': late_discrim > early_discrim,
                'per_segment_discrimination': discrimination.cpu().numpy().tolist(),
            }


class AblationTrainer:
    """Trainer for ablation study variants."""

    def __init__(
        self,
        model: AblationMHTPN,
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
                    'loss': f'{train_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'best': f'{self.best_val_accuracy:.4f}',
                })

            if patience_counter >= self.config.early_stopping_patience:
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'total_epochs': len(self.train_losses),
        }

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits, z_segments, segment_weights, _, _ = self.model.forward_with_explanations(inputs)

            ce_loss = self.criterion(logits, labels)

            # Prototype losses (with configured weights)
            cluster_loss, separation_loss, diversity_loss = self.model.compute_prototype_loss(
                z_segments, labels, segment_weights
            )

            loss = (
                ce_loss +
                self.config.cluster_loss_weight * cluster_loss +
                self.config.separation_loss_weight * separation_loss +
                self.config.diversity_loss_weight * diversity_loss
            )

            loss.backward()

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

    def evaluate(self, dataloader: DataLoader) -> EvaluationMetrics:
        """Evaluate the model and compute metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )

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


def compute_temporal_metrics(
    model: AblationMHTPN,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Compute temporal discrimination metrics."""
    model.eval()

    all_inputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            all_inputs.append(inputs)
            all_labels.append(labels)

    X = torch.cat(all_inputs, dim=0).to(device)
    y = torch.cat(all_labels, dim=0).to(device)

    return model.compute_temporal_discrimination(X, y)


def run_single_variant(
    config: AblationConfig,
    items: List,
    seq_len: int,
    input_dim: int,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single ablation variant with 5-fold cross-validation.

    Returns dictionary with all fold results and aggregate metrics.
    """
    set_all_seeds(RANDOM_SEED)

    X, y, patient_ids = extract_arrays(items)
    cv = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, patient_ids)):
        if verbose:
            print(f"    Fold {fold + 1}/{config.n_folds}...", flush=True)

        # Split data
        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        train_labels = np.array([item['label'] for item in train_items])

        # Create data loaders
        train_loader, val_loader, scaler = create_data_loaders(
            train_items, val_items, seq_len, config.batch_size
        )

        # Create model
        model = AblationMHTPN(
            input_dim=input_dim,
            num_classes=2,
            seq_len=seq_len,
            config=config,
        )

        # Train
        trainer = AblationTrainer(model, config, device)
        train_result = trainer.train(train_loader, val_loader, train_labels, verbose=False)

        # Evaluate
        metrics = trainer.evaluate(val_loader)

        # Compute temporal metrics
        temporal_metrics = compute_temporal_metrics(model, val_loader, device)

        # Get velocity norms (for ablation 2)
        velocity_info = model.get_velocity_norms()

        fold_result = {
            'fold': fold + 1,
            'accuracy': float(metrics.accuracy),
            'sensitivity': float(metrics.sensitivity),
            'specificity': float(metrics.specificity),
            'f1': float(metrics.f1),
            'auc_roc': float(metrics.auc_roc),
            'best_epoch': train_result['best_epoch'],
            'total_epochs': train_result['total_epochs'],
            'temporal': {
                'early_discrimination': temporal_metrics['early_discrimination'],
                'late_discrimination': temporal_metrics['late_discrimination'],
                'late_minus_early': temporal_metrics['late_minus_early'],
                'temporal_pattern_pass': temporal_metrics['temporal_pattern_pass'],
                'per_segment_discrimination': temporal_metrics['per_segment_discrimination'],
            },
            'velocity': velocity_info,
        }
        fold_results.append(fold_result)

        if verbose:
            tp = "PASS" if temporal_metrics['temporal_pattern_pass'] else "FAIL"
            print(f"      acc={metrics.accuracy:.1%}, temporal={tp}", flush=True)

    # Aggregate results
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    n_temporal_pass = sum(1 for r in fold_results if r['temporal']['temporal_pattern_pass'])

    mean_velocity = np.mean([r['velocity']['mean_velocity_norm'] for r in fold_results])

    # Per-segment discrimination averaged across folds
    n_segments = len(fold_results[0]['temporal']['per_segment_discrimination'])
    per_segment_avg = []
    for seg_idx in range(n_segments):
        seg_values = [r['temporal']['per_segment_discrimination'][seg_idx] for r in fold_results]
        per_segment_avg.append(float(np.mean(seg_values)))

    return {
        'config': config.to_dict(),
        'fold_results': fold_results,
        'aggregate': {
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'mean_sensitivity': float(np.mean([r['sensitivity'] for r in fold_results])),
            'mean_specificity': float(np.mean([r['specificity'] for r in fold_results])),
            'mean_f1': float(np.mean([r['f1'] for r in fold_results])),
            'mean_auc_roc': float(np.mean([r['auc_roc'] for r in fold_results])),
            'n_temporal_pass': n_temporal_pass,
            'temporal_pass_rate': f"{n_temporal_pass}/{config.n_folds}",
            'mean_velocity_norm': float(mean_velocity),
            'per_segment_discrimination_avg': per_segment_avg,
            'mean_late_minus_early': float(np.mean([r['temporal']['late_minus_early'] for r in fold_results])),
        },
    }


def run_ablation_study(
    output_dir: str,
    verbose: bool = True,
    ablation_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full ablation study with all 14 variants.

    Args:
        output_dir: Directory to save results
        verbose: Whether to print progress
        ablation_filter: Optional list of ablation names to run (e.g., ['n_segments', 'trajectory'])
                        If None, runs all ablations.

    Returns:
        Dictionary with all results
    """
    set_all_seeds(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'per_ablation'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 70, flush=True)
        print("MHTPN ABLATION STUDY", flush=True)
        print("=" * 70, flush=True)
        print(f"Device: {device}", flush=True)
        print(f"Output: {output_dir}", flush=True)

    # Load data once
    if verbose:
        print("\nLoading data...", flush=True)
    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)
    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]

    if verbose:
        print(f"Data: {len(items)} samples, seq_len={seq_len}, input_dim={input_dim}", flush=True)

    # Get all configs
    all_configs = get_all_ablation_configs()

    # Filter if specified
    if ablation_filter is not None:
        all_configs = {k: v for k, v in all_configs.items() if k in ablation_filter}

    # Count total runs
    total_variants = sum(len(configs) for configs in all_configs.values())
    total_runs = total_variants * 5  # 5 folds each

    if verbose:
        print(f"\nTotal variants: {total_variants}", flush=True)
        print(f"Total training runs: {total_runs}", flush=True)

    # Store all results
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'device': str(device),
            'seq_len': seq_len,
            'input_dim': input_dim,
            'n_samples': len(items),
            'total_variants': total_variants,
            'total_runs': total_runs,
        },
        'ablations': {},
    }

    variant_idx = 0

    for ablation_name, configs in all_configs.items():
        if verbose:
            print(f"\n{'='*60}", flush=True)
            print(f"ABLATION: {ablation_name} ({len(configs)} variants)", flush=True)
            print('='*60, flush=True)

        ablation_results = []

        for config in configs:
            variant_idx += 1

            if verbose:
                default_marker = " (default)" if config.is_default else ""
                print(f"\n  [{variant_idx}/{total_variants}] {config.ablation_id}: {config.variant_name}{default_marker}", flush=True)

            result = run_single_variant(
                config=config,
                items=items,
                seq_len=seq_len,
                input_dim=input_dim,
                device=device,
                verbose=verbose,
            )

            ablation_results.append(result)

            if verbose:
                acc = result['aggregate']['mean_accuracy']
                std = result['aggregate']['std_accuracy']
                tp_rate = result['aggregate']['temporal_pass_rate']
                print(f"    --> Accuracy: {acc:.1%} +/- {std:.1%}, Temporal: {tp_rate}", flush=True)

            # Save per-ablation results incrementally
            ablation_file = os.path.join(
                output_dir, 'per_ablation',
                f'{config.ablation_id}_{config.variant_name.replace(" ", "_").replace("+", "plus")}.json'
            )
            with open(ablation_file, 'w') as f:
                json.dump(result, f, indent=2)

        all_results['ablations'][ablation_name] = ablation_results

        # Save full results after each ablation completes
        full_results_path = os.path.join(output_dir, 'full_results.json')
        with open(full_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\n{'='*70}")
        print("ABLATION STUDY COMPLETE")
        print('='*70)
        print(f"\nResults saved to: {output_dir}")

    return all_results


def main():
    """Main entry point for running the ablation study."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/ccece/ablation_study_mhtpn/{timestamp}"

    run_ablation_study(output_dir, verbose=True)


if __name__ == '__main__':
    main()
