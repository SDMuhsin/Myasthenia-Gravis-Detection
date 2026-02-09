"""
MHTPN Diagnostic Script for LSST Dataset

Gathers hard evidence on why MHTPN underperforms on LSST:
1. Per-class accuracy breakdown
2. Confusion matrix
3. Training vs validation loss curves
4. Prediction distribution
5. Class probability distributions

Usage:
    python3 -m ccece.experiments.sota_comparison.mhtpn_diagnostics \
        --output results/ccece/sota_comparison/lsst_diagnostics
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.run_experiment import set_all_seeds

# Handle imports
try:
    from .datasets import load_dataset, get_cv_strategy, standardize_data
    from .mhtpn_configs import get_mhtpn_model_config, get_mhtpn_training_config
except ImportError:
    from ccece.experiments.sota_comparison.datasets import load_dataset, get_cv_strategy, standardize_data
    from ccece.experiments.sota_comparison.mhtpn_configs import get_mhtpn_model_config, get_mhtpn_training_config


RANDOM_SEED = 42


def run_diagnostic_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dataset_config,
    fold_idx: int,
    device: torch.device,
    output_dir: str,
) -> dict:
    """
    Run a single diagnostic fold with detailed logging.

    Returns dict with:
    - per_class_accuracy
    - confusion_matrix
    - predictions
    - probabilities
    - training_history (loss per epoch)
    """
    from ccece.models.multi_head_trajectory_proto_net import MultiHeadTrajectoryProtoNet

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    num_classes = dataset_config.n_classes

    # Get configs
    model_config = get_mhtpn_model_config(dataset_config.name, seq_len)
    train_config = get_mhtpn_training_config()

    # Set seed
    torch.manual_seed(RANDOM_SEED + fold_idx)
    np.random.seed(RANDOM_SEED + fold_idx)

    # Create model
    model = MultiHeadTrajectoryProtoNet(
        input_dim=input_dim,
        num_classes=num_classes,
        seq_len=seq_len,
        latent_dim=model_config.latent_dim,
        n_heads=model_config.n_heads,
        head_dim=model_config.head_dim,
        n_segments=model_config.n_segments,
        encoder_hidden=model_config.encoder_hidden,
        encoder_layers=model_config.encoder_layers,
        kernel_size=model_config.kernel_size,
        dropout=model_config.dropout,
    ).to(device)

    print(f"\n  Model config:")
    print(f"    latent_dim: {model_config.latent_dim}")
    print(f"    n_heads: {model_config.n_heads}")
    print(f"    n_segments: {model_config.n_segments}")
    print(f"    encoder_layers: {model_config.encoder_layers}")

    # Create data loaders
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)

    # Setup optimizer with class weights
    class_counts = np.bincount(y_train, minlength=num_classes)
    # Handle zero counts
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.from_numpy(class_weights).float().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config.epochs
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Training with detailed logging
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    pbar = tqdm(range(train_config.epochs), desc=f"Fold {fold_idx+1}", leave=True)

    for epoch in pbar:
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        scheduler.step()

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.3f}',
            'val_loss': f'{avg_val_loss:.3f}',
            'train_acc': f'{train_acc*100:.1f}%',
            'val_acc': f'{val_acc*100:.1f}%',
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= train_config.early_stopping_patience:
            print(f"\n    Early stopping at epoch {epoch + 1}")
            break

    # Load best model and get detailed predictions
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    # Compute per-class metrics
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class accuracy
    per_class_acc = {}
    for cls in range(num_classes):
        mask = y_true == cls
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == cls).sum() / mask.sum()
            per_class_acc[cls] = float(cls_acc)
        else:
            per_class_acc[cls] = None

    # Prediction distribution (what does model predict?)
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    true_counts = np.bincount(y_true, minlength=num_classes)

    return {
        'history': history,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': per_class_acc,
        'predictions': y_pred.tolist(),
        'probabilities': y_proba.tolist(),
        'true_labels': y_true.tolist(),
        'pred_distribution': pred_counts.tolist(),
        'true_distribution': true_counts.tolist(),
        'final_val_acc': float(best_val_acc),
        'class_weights': class_weights.tolist(),
    }


def plot_diagnostics(results: dict, output_dir: str, class_names: list):
    """Generate diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Training curves (average over folds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    for fold_idx, fold_result in enumerate(results['folds']):
        history = fold_result['history']
        ax.plot(history['train_loss'], alpha=0.3, color='blue')
        ax.plot(history['val_loss'], alpha=0.3, color='red')

    # Average
    max_epochs = max(len(f['history']['train_loss']) for f in results['folds'])
    avg_train_loss = np.zeros(max_epochs)
    avg_val_loss = np.zeros(max_epochs)
    count = np.zeros(max_epochs)

    for fold_result in results['folds']:
        history = fold_result['history']
        for i, (tl, vl) in enumerate(zip(history['train_loss'], history['val_loss'])):
            avg_train_loss[i] += tl
            avg_val_loss[i] += vl
            count[i] += 1

    avg_train_loss = avg_train_loss / np.maximum(count, 1)
    avg_val_loss = avg_val_loss / np.maximum(count, 1)

    ax.plot(avg_train_loss, color='blue', linewidth=2, label='Train (avg)')
    ax.plot(avg_val_loss, color='red', linewidth=2, label='Val (avg)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    avg_train_acc = np.zeros(max_epochs)
    avg_val_acc = np.zeros(max_epochs)
    count = np.zeros(max_epochs)

    for fold_result in results['folds']:
        history = fold_result['history']
        for i, (ta, va) in enumerate(zip(history['train_acc'], history['val_acc'])):
            avg_train_acc[i] += ta
            avg_val_acc[i] += va
            count[i] += 1

    avg_train_acc = avg_train_acc / np.maximum(count, 1)
    avg_val_acc = avg_val_acc / np.maximum(count, 1)

    ax.plot(avg_train_acc * 100, color='blue', linewidth=2, label='Train (avg)')
    ax.plot(avg_val_acc * 100, color='red', linewidth=2, label='Val (avg)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training vs Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Average confusion matrix
    n_classes = len(class_names)
    avg_cm = np.zeros((n_classes, n_classes))
    for fold_result in results['folds']:
        avg_cm += np.array(fold_result['confusion_matrix'])
    avg_cm = avg_cm / len(results['folds'])

    # Normalize by row (true labels)
    row_sums = avg_cm.sum(axis=1, keepdims=True)
    normalized_cm = avg_cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        normalized_cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix (avg over folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Per-class accuracy bar chart
    avg_per_class_acc = {}
    for cls in range(n_classes):
        accs = [f['per_class_accuracy'].get(str(cls), None) or f['per_class_accuracy'].get(cls, None)
                for f in results['folds']]
        valid_accs = [a for a in accs if a is not None]
        if valid_accs:
            avg_per_class_acc[cls] = np.mean(valid_accs)
        else:
            avg_per_class_acc[cls] = 0.0

    fig, ax = plt.subplots(figsize=(14, 6))
    classes = list(range(n_classes))
    accs = [avg_per_class_acc[c] * 100 for c in classes]

    colors = ['green' if a > 50 else 'orange' if a > 20 else 'red' for a in accs]
    bars = ax.bar([class_names[c] for c in classes], accs, color=colors)

    ax.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy (avg over folds)')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Prediction distribution vs true distribution
    avg_pred_dist = np.zeros(n_classes)
    avg_true_dist = np.zeros(n_classes)

    for fold_result in results['folds']:
        avg_pred_dist += np.array(fold_result['pred_distribution'])
        avg_true_dist += np.array(fold_result['true_distribution'])

    avg_pred_dist = avg_pred_dist / avg_pred_dist.sum() * 100
    avg_true_dist = avg_true_dist / avg_true_dist.sum() * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_classes)
    width = 0.35

    ax.bar(x - width/2, avg_true_dist, width, label='True Distribution', color='blue', alpha=0.7)
    ax.bar(x + width/2, avg_pred_dist, width, label='Predicted Distribution', color='orange', alpha=0.7)

    ax.set_xlabel('Class')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('True vs Predicted Class Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to: {output_dir}")


def run_diagnostics(
    dataset_name: str = 'LSST',
    output_dir: str = None,
    n_folds: int = 5,
):
    """Run full diagnostic analysis."""
    set_all_seeds(RANDOM_SEED)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/lsst_diagnostics_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print(f"MHTPN DIAGNOSTIC ANALYSIS: {dataset_name}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()

    # Load dataset
    print("Loading data...")
    X, y, groups, dataset_config = load_dataset(dataset_name, verbose=True)

    print(f"\nDataset Info:")
    print(f"  Samples: {len(y)}")
    print(f"  Seq length: {dataset_config.seq_len}")
    print(f"  Features: {dataset_config.n_features}")
    print(f"  Classes: {dataset_config.n_classes}")
    print(f"  Class names: {dataset_config.class_names}")

    # Class distribution
    class_counts = np.bincount(y, minlength=dataset_config.n_classes)
    print(f"\nClass distribution:")
    for i, (name, count) in enumerate(zip(dataset_config.class_names, class_counts)):
        pct = count / len(y) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")

    # Cross-validation
    cv = get_cv_strategy(dataset_config, n_splits=n_folds, random_state=RANDOM_SEED)

    if dataset_config.has_groups:
        splits = list(cv.split(X, y, groups))
    else:
        splits = list(cv.split(X, y))

    results = {
        'dataset': {
            'name': dataset_name,
            'n_samples': len(y),
            'seq_len': dataset_config.seq_len,
            'n_features': dataset_config.n_features,
            'n_classes': dataset_config.n_classes,
            'class_names': dataset_config.class_names,
            'class_counts': class_counts.tolist(),
        },
        'folds': [],
    }

    # Run each fold
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print('=' * 60)

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        X_train, X_val = standardize_data(X_train, X_val)

        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")

        fold_result = run_diagnostic_fold(
            X_train, y_train, X_val, y_val,
            dataset_config, fold_idx, device, output_dir
        )

        results['folds'].append(fold_result)

        print(f"\n  Results:")
        print(f"    Final val accuracy: {fold_result['final_val_acc']*100:.2f}%")

        # Print per-class accuracy
        print(f"    Per-class accuracy:")
        for cls, acc in fold_result['per_class_accuracy'].items():
            if acc is not None:
                cls_name = dataset_config.class_names[int(cls)] if int(cls) < len(dataset_config.class_names) else str(cls)
                status = "OK" if acc > 0.5 else "LOW" if acc > 0.2 else "ZERO"
                print(f"      {cls_name}: {acc*100:.1f}% [{status}]")

        # Save intermediate results
        fold_path = os.path.join(output_dir, f'fold{fold_idx+1}_results.json')
        with open(fold_path, 'w') as f:
            json.dump(fold_result, f, indent=2)

        torch.cuda.empty_cache()

    # Aggregate results
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print('=' * 70)

    val_accs = [f['final_val_acc'] for f in results['folds']]
    mean_acc = np.mean(val_accs)
    std_acc = np.std(val_accs)

    print(f"\nOverall Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # Average per-class accuracy
    print(f"\nAverage Per-Class Accuracy:")
    avg_per_class = {}
    for cls in range(dataset_config.n_classes):
        accs = []
        for f in results['folds']:
            acc = f['per_class_accuracy'].get(str(cls)) or f['per_class_accuracy'].get(cls)
            if acc is not None:
                accs.append(acc)
        if accs:
            avg_per_class[cls] = np.mean(accs)
            cls_name = dataset_config.class_names[cls] if cls < len(dataset_config.class_names) else str(cls)
            count = class_counts[cls]
            status = "OK" if avg_per_class[cls] > 0.5 else "LOW" if avg_per_class[cls] > 0.2 else "CRITICAL"
            print(f"  {cls_name:15s} (n={count:4d}): {avg_per_class[cls]*100:5.1f}% [{status}]")

    # Balanced accuracy
    balanced_acc = np.mean(list(avg_per_class.values()))
    print(f"\nBalanced Accuracy: {balanced_acc*100:.2f}%")
    print(f"Raw Accuracy: {mean_acc*100:.2f}%")
    print(f"Gap: {(mean_acc - balanced_acc)*100:.2f}% (higher = majority class bias)")

    # Save full results
    results['summary'] = {
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'balanced_accuracy': float(balanced_acc),
        'per_class_accuracy': {str(k): float(v) for k, v in avg_per_class.items()},
    }

    results_path = os.path.join(output_dir, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate plots
    print("\nGenerating diagnostic plots...")
    plot_diagnostics(results, output_dir, dataset_config.class_names)

    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"Results saved to: {output_dir}")
    print('=' * 70)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MHTPN Diagnostic Analysis')
    parser.add_argument('--dataset', type=str, default='LSST', help='Dataset name')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')

    args = parser.parse_args()

    run_diagnostics(
        dataset_name=args.dataset,
        output_dir=args.output,
        n_folds=args.folds,
    )
