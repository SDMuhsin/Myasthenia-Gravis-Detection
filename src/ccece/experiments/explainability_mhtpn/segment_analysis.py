"""
Component 2: Per-Segment Decision Analysis

Analyzes which temporal segments drive the classification decision
using MHTPN's intrinsic forward_with_explanations() outputs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader


def analyze_per_segment_decisions(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Analyze which segments drive predictions using intrinsic model outputs.

    Uses forward_with_explanations() - this is NOT post-hoc attribution.

    Args:
        model: Trained MHTPN model
        dataloader: DataLoader with validation data
        device: Torch device

    Returns:
        Dict with per-segment analysis results
    """
    model.eval()
    model = model.to(device)

    all_per_seg_sims = []  # Per-sample, per-segment similarities
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            # Get intrinsic explanations from model
            logits, z_segments, segment_weights, per_seg_sims_list, traj_sims_list = \
                model.forward_with_explanations(inputs)

            # Average per-segment similarities across heads
            # per_seg_sims_list is list of (batch, n_segments, n_classes) per head
            avg_sims = torch.stack(per_seg_sims_list, dim=0).mean(dim=0)  # (batch, n_seg, n_classes)

            all_per_seg_sims.append(avg_sims.cpu())
            all_labels.append(labels)

    # Concatenate all batches
    all_sims = torch.cat(all_per_seg_sims, dim=0)  # (N, n_segments, 2)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    n_samples, n_segments, n_classes = all_sims.shape

    # Split by true label
    hc_mask = (all_labels == 0)
    mg_mask = (all_labels == 1)

    # For each class, get similarity to correct prototype per segment
    # HC samples -> similarity to HC prototype (index 0)
    # MG samples -> similarity to MG prototype (index 1)
    hc_sims_to_correct = all_sims[hc_mask, :, 0]  # (n_hc, n_segments)
    mg_sims_to_correct = all_sims[mg_mask, :, 1]  # (n_mg, n_segments)

    # Also get similarity to wrong prototype
    hc_sims_to_wrong = all_sims[hc_mask, :, 1]  # (n_hc, n_segments)
    mg_sims_to_wrong = all_sims[mg_mask, :, 0]  # (n_mg, n_segments)

    # Mean across samples per segment
    hc_correct_mean = hc_sims_to_correct.mean(dim=0).numpy()  # (n_segments,)
    mg_correct_mean = mg_sims_to_correct.mean(dim=0).numpy()  # (n_segments,)
    hc_wrong_mean = hc_sims_to_wrong.mean(dim=0).numpy()
    mg_wrong_mean = mg_sims_to_wrong.mean(dim=0).numpy()

    # Standard deviations
    hc_correct_std = hc_sims_to_correct.std(dim=0).numpy()
    mg_correct_std = mg_sims_to_correct.std(dim=0).numpy()

    # Per-segment discrimination: how much more similar is each class to its own prototype?
    # Higher values = more discriminative
    discrimination = mg_correct_mean - hc_correct_mean

    # Early vs late analysis
    mid = n_segments // 2
    early_discrim = float(discrimination[:mid].mean())
    late_discrim = float(discrimination[mid:].mean())
    late_minus_early = late_discrim - early_discrim

    # Per-segment statistics
    per_segment_stats = []
    for seg_idx in range(n_segments):
        per_segment_stats.append({
            'segment': seg_idx + 1,
            'hc_correct_sim_mean': float(hc_correct_mean[seg_idx]),
            'hc_correct_sim_std': float(hc_correct_std[seg_idx]),
            'mg_correct_sim_mean': float(mg_correct_mean[seg_idx]),
            'mg_correct_sim_std': float(mg_correct_std[seg_idx]),
            'hc_wrong_sim_mean': float(hc_wrong_mean[seg_idx]),
            'mg_wrong_sim_mean': float(mg_wrong_mean[seg_idx]),
            'discrimination': float(discrimination[seg_idx]),
            'hc_margin': float(hc_correct_mean[seg_idx] - hc_wrong_mean[seg_idx]),
            'mg_margin': float(mg_correct_mean[seg_idx] - mg_wrong_mean[seg_idx]),
        })

    # Determine most discriminative segment
    most_discrim_segment = int(discrimination.argmax()) + 1
    segment_ranking = (np.argsort(discrimination)[::-1] + 1).tolist()

    # Clinical interpretation
    if late_discrim > early_discrim:
        clinical_interpretation = "Late segments show higher discrimination, consistent with MG fatigability hypothesis"
    else:
        clinical_interpretation = "Early segments show higher discrimination, inconsistent with MG fatigability hypothesis"

    results = {
        'per_segment_discrimination': discrimination.tolist(),
        'per_segment_stats': per_segment_stats,
        'hc_mean_similarity_profile': hc_correct_mean.tolist(),
        'mg_mean_similarity_profile': mg_correct_mean.tolist(),
        'n_segments': n_segments,
        'early_segments': list(range(1, mid + 1)),
        'late_segments': list(range(mid + 1, n_segments + 1)),
        'early_discrimination': early_discrim,
        'late_discrimination': late_discrim,
        'late_minus_early': late_minus_early,
        'temporal_pattern_pass': bool(late_discrim > early_discrim),
        'most_discriminative_segment': most_discrim_segment,
        'segment_discrimination_ranking': segment_ranking,
        'clinical_interpretation': clinical_interpretation,
        'n_samples': {
            'total': int(n_samples),
            'hc': int(hc_mask.sum().item()),
            'mg': int(mg_mask.sum().item()),
        },
        'success_criteria': {
            'temporal_pattern_pass': {
                'threshold': 'late_discrim > early_discrim',
                'early_value': early_discrim,
                'late_value': late_discrim,
                'passed': bool(late_discrim > early_discrim),
            },
            'late_minus_early_positive': {
                'threshold': '> 0',
                'value': late_minus_early,
                'passed': late_minus_early > 0,
            },
            'most_discrim_in_late_half': {
                'threshold': 'segment >= mid',
                'value': most_discrim_segment,
                'mid': mid,
                'passed': most_discrim_segment > mid,
            },
        },
    }

    return results


def analyze_per_head_segment_patterns(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Analyze per-segment patterns for each head separately.

    Shows whether different heads focus on different temporal patterns.
    """
    model.eval()
    model = model.to(device)

    # Collect per-head, per-segment similarities
    n_heads = model.n_heads
    head_sims = [[] for _ in range(n_heads)]
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            logits, z_segments, segment_weights, per_seg_sims_list, traj_sims_list = \
                model.forward_with_explanations(inputs)

            for head_idx, per_seg_sims in enumerate(per_seg_sims_list):
                head_sims[head_idx].append(per_seg_sims.cpu())

            all_labels.append(labels)

    all_labels = torch.cat(all_labels, dim=0)
    hc_mask = (all_labels == 0)
    mg_mask = (all_labels == 1)

    results = {}
    late_minus_early_values = []

    for head_idx in range(n_heads):
        head_sims_all = torch.cat(head_sims[head_idx], dim=0)  # (N, n_seg, 2)
        n_segments = head_sims_all.size(1)

        # Per-class similarities to correct prototype
        hc_correct = head_sims_all[hc_mask, :, 0]
        mg_correct = head_sims_all[mg_mask, :, 1]

        hc_mean = hc_correct.mean(dim=0).numpy()
        mg_mean = mg_correct.mean(dim=0).numpy()

        discrimination = mg_mean - hc_mean

        mid = n_segments // 2
        early_discrim = float(discrimination[:mid].mean())
        late_discrim = float(discrimination[mid:].mean())
        late_minus_early = late_discrim - early_discrim

        peak_segment = int(discrimination.argmax()) + 1

        results[f'head_{head_idx}'] = {
            'per_segment_discrimination': discrimination.tolist(),
            'early_discrimination': early_discrim,
            'late_discrimination': late_discrim,
            'late_minus_early': late_minus_early,
            'temporal_pattern_pass': bool(late_discrim > early_discrim),
            'peak_segment': peak_segment,
            'hc_similarity_profile': hc_mean.tolist(),
            'mg_similarity_profile': mg_mean.tolist(),
        }

        late_minus_early_values.append(late_minus_early)

    # Aggregate across heads
    n_pass = sum(1 for r in results.values() if isinstance(r, dict) and r.get('temporal_pattern_pass', False))

    results['aggregate'] = {
        'head_agreement_on_late_pattern': n_pass,
        'heads_with_late_gt_early': n_pass,
        'n_heads': n_heads,
        'all_heads_pass': n_pass == n_heads,
        'mean_late_minus_early': float(np.mean(late_minus_early_values)),
        'std_late_minus_early': float(np.std(late_minus_early_values)),
    }

    return results
