"""
Component 3: Head Velocity Diversity Analysis

Analyzes diversity of learned velocities across heads.
MHTPN is trained with diversity loss that encourages orthogonal velocities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from sklearn.decomposition import PCA


def analyze_velocity_diversity(model, device: torch.device) -> Dict[str, Any]:
    """
    Analyze diversity of learned velocities across heads.

    The model is trained with diversity_loss that encourages orthogonal velocities.

    Args:
        model: Trained MHTPN model
        device: Torch device

    Returns:
        Dict with velocity diversity analysis
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # Collect all velocities: (n_heads * 2, head_dim)
        all_velocities = []
        velocity_labels = []  # (head_idx, class_name)

        for head_idx, head in enumerate(model.heads):
            all_velocities.append(head.prototype_velocities[0].detach().cpu())  # HC
            all_velocities.append(head.prototype_velocities[1].detach().cpu())  # MG
            velocity_labels.append((head_idx, 'HC'))
            velocity_labels.append((head_idx, 'MG'))

        velocities = torch.stack(all_velocities)  # (n_heads*2, head_dim)

        # Normalize for cosine similarity
        velocities_norm = F.normalize(velocities, dim=1)

        # Cosine similarity matrix
        cosine_matrix = torch.mm(velocities_norm, velocities_norm.t())

        # Off-diagonal statistics (excluding self-similarity)
        n = cosine_matrix.size(0)
        mask = ~torch.eye(n, dtype=torch.bool)
        off_diag = cosine_matrix[mask]

        # Per-head velocity norms
        head_velocity_norms = []
        for head_idx, head in enumerate(model.heads):
            hc_norm = torch.norm(head.prototype_velocities[0]).item()
            mg_norm = torch.norm(head.prototype_velocities[1]).item()
            head_velocity_norms.append({
                'head': head_idx,
                'hc_norm': hc_norm,
                'mg_norm': mg_norm,
                'mean_norm': (hc_norm + mg_norm) / 2,
            })

        # Velocity magnitudes for all
        all_norms = velocities.norm(dim=1).numpy()

        # Diversity score: 1 - mean absolute off-diagonal similarity
        diversity_score = 1.0 - off_diag.abs().mean().item()

        # Create labeled cosine matrix
        labels = [f'H{h}-{c}' for h, c in velocity_labels]
        cosine_matrix_labeled = {
            'matrix': cosine_matrix.numpy().tolist(),
            'labels': labels,
        }

        # Separate HC-HC, MG-MG, and HC-MG similarities
        hc_indices = [i for i, (_, c) in enumerate(velocity_labels) if c == 'HC']
        mg_indices = [i for i, (_, c) in enumerate(velocity_labels) if c == 'MG']

        hc_hc_sims = []
        mg_mg_sims = []
        hc_mg_sims = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_matrix[i, j].item()
                if i in hc_indices and j in hc_indices:
                    hc_hc_sims.append(sim)
                elif i in mg_indices and j in mg_indices:
                    mg_mg_sims.append(sim)
                else:
                    hc_mg_sims.append(sim)

        # PCA of velocities
        pca = PCA(n_components=2)
        velocities_2d = pca.fit_transform(velocities.numpy())

        velocity_pca = {
            'coordinates': velocities_2d.tolist(),
            'labels': labels,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
        }

        # Compute centroids by class
        hc_coords = velocities_2d[hc_indices]
        mg_coords = velocities_2d[mg_indices]

        velocity_pca['hc_centroid'] = hc_coords.mean(axis=0).tolist()
        velocity_pca['mg_centroid'] = mg_coords.mean(axis=0).tolist()
        velocity_pca['class_separation'] = float(np.linalg.norm(
            np.array(velocity_pca['hc_centroid']) - np.array(velocity_pca['mg_centroid'])
        ))

    results = {
        'cosine_similarity_matrix': cosine_matrix_labeled,
        'mean_off_diagonal_similarity': float(off_diag.mean().item()),
        'std_off_diagonal_similarity': float(off_diag.std().item()),
        'max_off_diagonal_similarity': float(off_diag.max().item()),
        'min_off_diagonal_similarity': float(off_diag.min().item()),
        'diversity_score': diversity_score,
        'velocity_norms_per_head': head_velocity_norms,
        'all_velocity_norms': all_norms.tolist(),
        'mean_velocity_norm': float(np.mean(all_norms)),
        'std_velocity_norm': float(np.std(all_norms)),
        'min_velocity_norm': float(np.min(all_norms)),
        'velocity_pca': velocity_pca,
        'within_class_similarity': {
            'hc_hc_mean': float(np.mean(hc_hc_sims)) if hc_hc_sims else None,
            'mg_mg_mean': float(np.mean(mg_mg_sims)) if mg_mg_sims else None,
            'hc_hc_values': hc_hc_sims,
            'mg_mg_values': mg_mg_sims,
        },
        'between_class_similarity': {
            'hc_mg_mean': float(np.mean(hc_mg_sims)) if hc_mg_sims else None,
            'hc_mg_values': hc_mg_sims,
        },
        'interpretation': _interpret_diversity(diversity_score, float(off_diag.mean().item())),
        'success_criteria': {
            'diversity_score_high': {
                'threshold': '> 0.5',
                'value': diversity_score,
                'passed': diversity_score > 0.5,
            },
            'mean_off_diag_low': {
                'threshold': '< 0.6',
                'value': float(off_diag.mean().item()),
                'passed': abs(float(off_diag.mean().item())) < 0.6,
            },
            'all_velocities_nonzero': {
                'threshold': 'min norm > 0.05',
                'value': float(np.min(all_norms)),
                'passed': float(np.min(all_norms)) > 0.05,
            },
        },
    }

    return results


def _interpret_diversity(diversity_score: float, mean_off_diag: float) -> str:
    """Generate interpretation of diversity metrics."""
    if diversity_score > 0.7:
        diversity_level = "high"
    elif diversity_score > 0.5:
        diversity_level = "moderate"
    else:
        diversity_level = "low"

    if abs(mean_off_diag) < 0.3:
        similarity_level = "nearly orthogonal"
    elif abs(mean_off_diag) < 0.6:
        similarity_level = "moderately diverse"
    else:
        similarity_level = "somewhat correlated"

    return f"Heads show {diversity_level} diversity (score={diversity_score:.2f}). " \
           f"Velocity vectors are {similarity_level} (mean cosine={mean_off_diag:.2f})."


def compute_per_head_velocity_analysis(model, device: torch.device) -> Dict[str, Any]:
    """
    Detailed per-head velocity analysis.

    Returns breakdown of velocity characteristics for each head.
    """
    model.eval()
    model = model.to(device)

    results = {}

    with torch.no_grad():
        for head_idx, head in enumerate(model.heads):
            hc_velocity = head.prototype_velocities[0].detach().cpu()
            mg_velocity = head.prototype_velocities[1].detach().cpu()

            hc_norm = torch.norm(hc_velocity).item()
            mg_norm = torch.norm(mg_velocity).item()

            # Cosine similarity between HC and MG velocities for this head
            within_head_cosine = F.cosine_similarity(
                hc_velocity.unsqueeze(0),
                mg_velocity.unsqueeze(0)
            ).item()

            # Velocity direction analysis
            hc_dominant_dims = torch.argsort(hc_velocity.abs(), descending=True)[:3].numpy().tolist()
            mg_dominant_dims = torch.argsort(mg_velocity.abs(), descending=True)[:3].numpy().tolist()

            results[f'head_{head_idx}'] = {
                'hc_velocity_norm': hc_norm,
                'mg_velocity_norm': mg_norm,
                'within_head_cosine_sim': within_head_cosine,
                'velocities_opposed': within_head_cosine < 0,
                'velocities_orthogonal': abs(within_head_cosine) < 0.3,
                'hc_dominant_dimensions': hc_dominant_dims,
                'mg_dominant_dimensions': mg_dominant_dims,
                'dimension_overlap': len(set(hc_dominant_dims) & set(mg_dominant_dims)),
            }

    # Aggregate
    within_head_cosines = [r['within_head_cosine_sim'] for k, r in results.items() if k.startswith('head_')]
    n_opposed = sum(1 for c in within_head_cosines if c < 0)
    n_orthogonal = sum(1 for c in within_head_cosines if abs(c) < 0.3)

    results['aggregate'] = {
        'mean_within_head_cosine': float(np.mean(within_head_cosines)),
        'n_heads_with_opposed_velocities': n_opposed,
        'n_heads_with_orthogonal_velocities': n_orthogonal,
        'n_heads': model.n_heads,
    }

    return results
