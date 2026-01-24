"""
CCECE Paper: Generate Real Explainability Figures

Trains a CNN1D model and generates actual explainability visualizations.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ccece.data_loader import load_binary_dataset, prepare_cv_splits, subsample_sequence, pad_or_truncate, add_engineered_features
from ccece.models import get_model
from ccece.trainer import Trainer, TrainingConfig
from ccece.explainability import ModelExplainer, GradientSaliency

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def main():
    output_dir = './results/ccece/figures'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    items = load_binary_dataset(verbose=False)
    splits = list(prepare_cv_splits(items, n_splits=5, random_state=42))
    train_items, val_items = splits[0]

    seq_len = 2903

    # Prepare data with engineered features
    def prepare_data(data_items):
        X_list = []
        y_list = []
        for item in data_items:
            x = add_engineered_features(item['data'])  # 6 -> 14 features
            x = subsample_sequence(x, factor=10)
            x = pad_or_truncate(x, seq_len)
            X_list.append(x)
            y_list.append(item['label'])
        return np.array(X_list), np.array(y_list)

    X_train, y_train = prepare_data(train_items)
    X_val, y_val = prepare_data(val_items)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # Create and train model (quick training)
    print("Training CNN1D model (10 epochs)...")
    model = get_model(
        name='cnn1d',
        input_dim=14,
        num_classes=2,
        seq_len=seq_len,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        fc_dim=64,
    )

    config = TrainingConfig(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=5,
    )

    trainer = Trainer(model, config, device)
    trainer.train(train_loader, val_loader, y_train)

    # Create explainer
    print("\nGenerating explainability visualizations...")
    model.eval()
    explainer = ModelExplainer(model, device)

    # Get sample from validation set
    # Find one MG and one HC sample
    mg_sample = None
    hc_sample = None

    for X, y in val_loader:
        for i in range(len(y)):
            if y[i].item() == 1 and mg_sample is None:
                mg_sample = (X[i:i+1], y[i].item())
            elif y[i].item() == 0 and hc_sample is None:
                hc_sample = (X[i:i+1], y[i].item())

            if mg_sample is not None and hc_sample is not None:
                break
        if mg_sample is not None and hc_sample is not None:
            break

    # Generate explanations
    samples = [('MG', mg_sample), ('HC', hc_sample)]

    for label_name, (X, true_label) in samples:
        print(f"\nGenerating explanation for {label_name} sample...")
        result = explainer.explain(X, label=true_label, method='integrated')

        # Feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_names = explainer.get_feature_names()
        importance = result.feature_importance

        sorted_idx = np.argsort(importance)
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(feature_names)))

        ax.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx] * 100,
                color=colors)
        ax.set_xlabel('Feature Importance (%)')
        ax.set_title(f'Feature Importance for {label_name} Sample\n'
                     f'(Predicted: {"MG" if result.predicted_label == 1 else "HC"}, '
                     f'Confidence: {result.predicted_probability:.1%})')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_importance_{label_name.lower()}.png'))
        plt.savefig(os.path.join(output_dir, f'feature_importance_{label_name.lower()}.pdf'))
        plt.close()

        # Temporal saliency plot
        fig, ax = plt.subplots(figsize=(12, 4))
        time_steps = len(result.temporal_saliency)
        t = np.linspace(0, time_steps / 100, time_steps)  # 100 Hz after 10x subsampling

        ax.fill_between(t, result.temporal_saliency, alpha=0.6, color='steelblue')
        ax.plot(t, result.temporal_saliency, color='navy', linewidth=0.5)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Temporal Saliency')
        ax.set_title(f'Temporal Importance for {label_name} Sample')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'temporal_saliency_{label_name.lower()}.png'))
        plt.savefig(os.path.join(output_dir, f'temporal_saliency_{label_name.lower()}.pdf'))
        plt.close()

        # Saliency heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        saliency_map = result.saliency_map.T  # (features, seq_len)

        # Subsample for visualization
        subsample = max(1, saliency_map.shape[1] // 500)
        saliency_subsampled = saliency_map[:, ::subsample]

        im = ax.imshow(saliency_subsampled, aspect='auto', cmap='hot',
                       interpolation='bilinear')
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Time (subsampled)')
        ax.set_ylabel('Feature')
        ax.set_title(f'Saliency Heatmap for {label_name} Sample')
        plt.colorbar(im, ax=ax, label='Saliency')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'saliency_heatmap_{label_name.lower()}.png'))
        plt.savefig(os.path.join(output_dir, f'saliency_heatmap_{label_name.lower()}.pdf'))
        plt.close()

        print(f"  - Saved feature_importance_{label_name.lower()}.png/pdf")
        print(f"  - Saved temporal_saliency_{label_name.lower()}.png/pdf")
        print(f"  - Saved saliency_heatmap_{label_name.lower()}.png/pdf")

        # Print summary
        print(f"\n  Top 5 features for {label_name}:")
        for idx, score in result.get_top_features(5):
            print(f"    {feature_names[idx]}: {score:.4f}")

    # Combined comparison figure
    print("\nGenerating combined comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, (label_name, (X, true_label)) in enumerate(samples):
        result = explainer.explain(X, label=true_label, method='integrated')

        # Feature importance
        ax = axes[i, 0]
        feature_names = explainer.get_feature_names()
        importance = result.feature_importance
        sorted_idx = np.argsort(importance)[-8:]  # Top 8

        ax.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx] * 100,
                color='steelblue' if label_name == 'MG' else 'coral')
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'{label_name} Sample - Top 8 Features')

        # Temporal saliency
        ax = axes[i, 1]
        t = np.linspace(0, len(result.temporal_saliency) / 100, len(result.temporal_saliency))
        ax.fill_between(t, result.temporal_saliency, alpha=0.6,
                        color='steelblue' if label_name == 'MG' else 'coral')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Saliency')
        ax.set_title(f'{label_name} Sample - Temporal Saliency')

    plt.suptitle('Explainability Comparison: MG vs HC', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'explainability_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'explainability_comparison.pdf'))
    plt.close()

    print("\nSaved: explainability_comparison.png/pdf")
    print("\n" + "=" * 60)
    print(f"All explainability figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
