"""
Component 4: Sample-Level Trajectory Analysis (Case Studies)

Analyzes how individual samples relate to the learned prototype trajectories.
Provides clinically interpretable explanations for individual predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader
from scipy import stats


def analyze_single_sample(
    model,
    x: torch.Tensor,
    true_label: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Analyze how a single sample's encoding relates to prototype trajectories.

    Returns per-segment similarities showing when/why the model makes its decision.

    Args:
        model: Trained MHTPN model
        x: Input tensor (seq_len, input_dim) or (1, seq_len, input_dim)
        true_label: Ground truth label (0=HC, 1=MG)
        device: Torch device

    Returns:
        Dict with sample-level analysis
    """
    model.eval()

    with torch.no_grad():
        # Ensure batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.to(device)

        # Get explanations
        logits, z_segments, segment_weights, per_seg_sims_list, traj_sims_list = \
            model.forward_with_explanations(x)

        # Average across heads
        avg_sims = torch.stack(per_seg_sims_list, dim=0).mean(dim=0)  # (1, n_seg, 2)
        avg_sims = avg_sims[0]  # (n_seg, 2)

        n_segments = avg_sims.size(0)

        hc_sims = avg_sims[:, 0].cpu().numpy()
        mg_sims = avg_sims[:, 1].cpu().numpy()

        # Gap: positive means closer to MG prototype
        gap = mg_sims - hc_sims

        # Linear regression for trend
        t = np.arange(len(gap))
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, gap)

        # Prediction
        probs = F.softmax(logits[0], dim=0).cpu().numpy()
        pred = int(probs.argmax())
        confidence = float(probs.max())
        margin = float(probs[1] - probs[0])  # Positive = toward MG

        # Per-head analysis
        per_head_predictions = []
        for head_idx, per_seg_sims in enumerate(per_seg_sims_list):
            head_sims = per_seg_sims[0]  # (n_seg, 2)
            head_traj_sims = traj_sims_list[head_idx][0]  # (2,)
            head_pred = int(head_traj_sims.argmax().item())
            per_head_predictions.append(head_pred)

        head_agreement = sum(1 for p in per_head_predictions if p == pred) / len(per_head_predictions)

        # Decision segment: first segment where gap clearly favors the prediction
        threshold = 0.1
        decision_segment = -1
        expected_sign = 1 if pred == 1 else -1
        for seg_idx in range(n_segments):
            if gap[seg_idx] * expected_sign > threshold:
                decision_segment = seg_idx + 1
                break

        # Clinical interpretation
        if pred == true_label:
            correctness = "correct"
        else:
            correctness = "incorrect"

        if pred == 1:  # Predicted MG
            if slope > 0.05:
                trend_interpretation = "Increasing similarity to MG over time (consistent with fatigability)"
            elif slope < -0.05:
                trend_interpretation = "Decreasing similarity to MG over time (inconsistent)"
            else:
                trend_interpretation = "Stable similarity pattern"
        else:  # Predicted HC
            if slope < -0.05:
                trend_interpretation = "Increasing similarity to HC over time"
            elif slope > 0.05:
                trend_interpretation = "Decreasing similarity to HC over time (inconsistent)"
            else:
                trend_interpretation = "Stable similarity pattern"

    return {
        'true_label': true_label,
        'true_label_name': 'MG' if true_label == 1 else 'HC',
        'predicted_label': pred,
        'predicted_label_name': 'MG' if pred == 1 else 'HC',
        'correct': pred == true_label,
        'confidence': confidence,
        'margin': margin,
        'hc_probability': float(probs[0]),
        'mg_probability': float(probs[1]),
        'per_segment_hc_similarity': hc_sims.tolist(),
        'per_segment_mg_similarity': mg_sims.tolist(),
        'similarity_gap': gap.tolist(),
        'gap_trend_slope': float(slope),
        'gap_trend_intercept': float(intercept),
        'gap_trend_r_squared': float(r_value ** 2),
        'gap_trend_p_value': float(p_value),
        'gap_trend_significant': p_value < 0.05,
        'decision_segment': decision_segment,
        'per_head_predictions': per_head_predictions,
        'head_agreement': head_agreement,
        'all_heads_agree': head_agreement == 1.0,
        'trend_interpretation': trend_interpretation,
        'correctness': correctness,
    }


def select_case_study_samples(
    model,
    dataloader: DataLoader,
    device: torch.device,
    n_per_category: int = 5,
) -> Dict[str, List[Tuple[int, torch.Tensor, int]]]:
    """
    Select samples for case studies based on prediction characteristics.

    Categories:
    - high_conf_correct_mg: High confidence correct MG predictions
    - high_conf_correct_hc: High confidence correct HC predictions
    - low_conf_correct: Low confidence but correct predictions
    - misclassified: Incorrect predictions

    Returns:
        Dict mapping category to list of (sample_idx, x, true_label)
    """
    model.eval()

    all_samples = []
    sample_idx = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = F.softmax(logits, dim=1).cpu()

            for i in range(inputs.size(0)):
                x = inputs[i].cpu()
                true_label = int(labels[i].item())
                pred = int(probs[i].argmax().item())
                confidence = float(probs[i].max().item())

                all_samples.append({
                    'idx': sample_idx,
                    'x': x,
                    'true_label': true_label,
                    'pred': pred,
                    'confidence': confidence,
                    'correct': pred == true_label,
                    'mg_prob': float(probs[i, 1].item()),
                    'hc_prob': float(probs[i, 0].item()),
                })
                sample_idx += 1

    # Select samples for each category
    categories = {
        'high_conf_correct_mg': [],
        'high_conf_correct_hc': [],
        'low_conf_correct': [],
        'misclassified_mg_to_hc': [],
        'misclassified_hc_to_mg': [],
    }

    # High confidence correct MG
    mg_correct = [s for s in all_samples if s['true_label'] == 1 and s['correct']]
    mg_correct.sort(key=lambda s: s['confidence'], reverse=True)
    categories['high_conf_correct_mg'] = mg_correct[:n_per_category]

    # High confidence correct HC
    hc_correct = [s for s in all_samples if s['true_label'] == 0 and s['correct']]
    hc_correct.sort(key=lambda s: s['confidence'], reverse=True)
    categories['high_conf_correct_hc'] = hc_correct[:n_per_category]

    # Low confidence correct (either class)
    all_correct = [s for s in all_samples if s['correct']]
    all_correct.sort(key=lambda s: s['confidence'])
    categories['low_conf_correct'] = all_correct[:n_per_category]

    # Misclassified MG as HC
    mg_to_hc = [s for s in all_samples if s['true_label'] == 1 and s['pred'] == 0]
    mg_to_hc.sort(key=lambda s: s['confidence'], reverse=True)
    categories['misclassified_mg_to_hc'] = mg_to_hc[:n_per_category]

    # Misclassified HC as MG
    hc_to_mg = [s for s in all_samples if s['true_label'] == 0 and s['pred'] == 1]
    hc_to_mg.sort(key=lambda s: s['confidence'], reverse=True)
    categories['misclassified_hc_to_mg'] = hc_to_mg[:n_per_category]

    # Convert to tuples
    result = {}
    for cat_name, samples in categories.items():
        result[cat_name] = [(s['idx'], s['x'], s['true_label']) for s in samples]

    return result


def analyze_case_studies(
    model,
    dataloader: DataLoader,
    device: torch.device,
    n_per_category: int = 5,
) -> Dict[str, Any]:
    """
    Complete case study analysis.

    Args:
        model: Trained MHTPN model
        dataloader: DataLoader with validation data
        device: Torch device
        n_per_category: Number of samples per category

    Returns:
        Dict with all case study results and statistics
    """
    model.eval()

    # Select samples
    selected = select_case_study_samples(model, dataloader, device, n_per_category)

    # Analyze each sample
    case_studies = {}
    all_slopes = []
    category_slopes = {}

    for cat_name, samples in selected.items():
        case_studies[cat_name] = []
        category_slopes[cat_name] = []

        for sample_idx, x, true_label in samples:
            analysis = analyze_single_sample(model, x, true_label, device)
            analysis['sample_idx'] = sample_idx
            case_studies[cat_name].append(analysis)
            all_slopes.append(analysis['gap_trend_slope'])
            category_slopes[cat_name].append(analysis['gap_trend_slope'])

    # Compute statistics across categories
    gap_trend_statistics = {}

    for cat_name, slopes in category_slopes.items():
        if slopes:
            gap_trend_statistics[cat_name] = {
                'mean_slope': float(np.mean(slopes)),
                'std_slope': float(np.std(slopes)),
                'min_slope': float(np.min(slopes)),
                'max_slope': float(np.max(slopes)),
                'n_positive_slope': int(sum(1 for s in slopes if s > 0)),
                'n_samples': len(slopes),
            }

    # Statistical test: MG correct vs HC correct slopes
    mg_slopes = category_slopes.get('high_conf_correct_mg', [])
    hc_slopes = category_slopes.get('high_conf_correct_hc', [])

    if len(mg_slopes) >= 2 and len(hc_slopes) >= 2:
        t_stat, p_value = stats.ttest_ind(mg_slopes, hc_slopes)
        cohens_d = (np.mean(mg_slopes) - np.mean(hc_slopes)) / np.sqrt(
            (np.var(mg_slopes) + np.var(hc_slopes)) / 2
        )
    else:
        t_stat, p_value, cohens_d = None, None, None

    gap_trend_statistics['mg_vs_hc_comparison'] = {
        't_statistic': float(t_stat) if t_stat is not None else None,
        'p_value': float(p_value) if p_value is not None else None,
        'cohens_d': float(cohens_d) if cohens_d is not None else None,
        'significant': p_value < 0.05 if p_value is not None else None,
    }

    # Decision segment statistics
    decision_segments = []
    for cat_name, studies in case_studies.items():
        for study in studies:
            if study['decision_segment'] > 0:
                decision_segments.append(study['decision_segment'])

    n_segments = model.n_segments
    mid = n_segments // 2

    if decision_segments:
        decision_segment_stats = {
            'mean_decision_segment': float(np.mean(decision_segments)),
            'std_decision_segment': float(np.std(decision_segments)),
            'late_decision_fraction': float(sum(1 for d in decision_segments if d > mid) / len(decision_segments)),
            'distribution': {i: decision_segments.count(i) for i in range(1, n_segments + 1)},
        }
    else:
        decision_segment_stats = {
            'mean_decision_segment': None,
            'late_decision_fraction': None,
            'distribution': {},
        }

    # Head agreement statistics
    head_agreements = []
    for cat_name, studies in case_studies.items():
        for study in studies:
            head_agreements.append(study['head_agreement'])

    head_agreement_stats = {
        'mean_agreement': float(np.mean(head_agreements)),
        'n_unanimous': int(sum(1 for a in head_agreements if a == 1.0)),
        'total_samples': len(head_agreements),
    }

    # Success criteria
    success_criteria = {
        'mg_correct_positive_slope': {
            'description': 'MG correct samples should have positive gap slope (toward MG)',
            'value': gap_trend_statistics.get('high_conf_correct_mg', {}).get('mean_slope'),
            'threshold': '> 0',
            'passed': gap_trend_statistics.get('high_conf_correct_mg', {}).get('mean_slope', 0) > 0,
        },
        'hc_correct_negative_slope': {
            'description': 'HC correct samples should have negative gap slope (toward HC)',
            'value': gap_trend_statistics.get('high_conf_correct_hc', {}).get('mean_slope'),
            'threshold': '< 0',
            'passed': gap_trend_statistics.get('high_conf_correct_hc', {}).get('mean_slope', 0) < 0,
        },
        'slope_difference_significant': {
            'description': 'MG vs HC slope difference should be significant',
            'p_value': gap_trend_statistics['mg_vs_hc_comparison'].get('p_value'),
            'threshold': 'p < 0.05',
            'passed': gap_trend_statistics['mg_vs_hc_comparison'].get('significant', False),
        },
    }

    return {
        'case_studies': case_studies,
        'gap_trend_statistics': gap_trend_statistics,
        'decision_segment_stats': decision_segment_stats,
        'head_agreement_stats': head_agreement_stats,
        'success_criteria': success_criteria,
        'summary': {
            'total_samples_analyzed': sum(len(studies) for studies in case_studies.values()),
            'n_categories': len(case_studies),
            'mg_mean_slope': gap_trend_statistics.get('high_conf_correct_mg', {}).get('mean_slope'),
            'hc_mean_slope': gap_trend_statistics.get('high_conf_correct_hc', {}).get('mean_slope'),
            'slope_test_significant': success_criteria['slope_difference_significant']['passed'],
        },
    }
