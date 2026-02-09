"""
Component 1: Trajectory Prototype Analysis

Analyzes the learned trajectory prototypes in MHTPN:
- Prototype origins and velocities
- Trajectory evolution over time
- Inter-class distance dynamics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from sklearn.decomposition import PCA


def compute_trajectory_evolution(model, n_points: int = 20) -> Dict[str, Any]:
    """
    Compute prototype positions at multiple time points.

    Args:
        model: Trained MHTPN model
        n_points: Number of time points to compute

    Returns:
        Dict with per-head trajectory data and aggregate metrics
    """
    t_values = torch.linspace(0, 1, n_points)
    results = {}

    all_hc_velocities = []
    all_mg_velocities = []
    all_origin_separations = []
    all_velocity_cosines = []
    all_trajectory_divergences = []

    for head_idx, head in enumerate(model.heads):
        # Get prototype positions at each time point
        positions = []
        for t in t_values:
            # p(t) = origin + t * velocity
            t_tensor = torch.tensor([t], device=head.prototype_origins.device)
            pos = head.prototype_origins + t_tensor.unsqueeze(1) * head.prototype_velocities
            positions.append(pos.detach().cpu())

        positions = torch.stack(positions)  # (n_points, 2, head_dim)

        # HC trajectory (class 0) and MG trajectory (class 1)
        hc_trajectory = positions[:, 0, :]  # (n_points, head_dim)
        mg_trajectory = positions[:, 1, :]  # (n_points, head_dim)

        # Compute inter-class distance over time
        inter_class_dist = torch.norm(mg_trajectory - hc_trajectory, dim=1)

        # Velocity norms
        hc_velocity_norm = torch.norm(head.prototype_velocities[0]).item()
        mg_velocity_norm = torch.norm(head.prototype_velocities[1]).item()

        # Origin separation
        origin_separation = torch.dist(
            head.prototype_origins[0],
            head.prototype_origins[1]
        ).item()

        # Velocity cosine similarity
        velocity_cosine = F.cosine_similarity(
            head.prototype_velocities[0].unsqueeze(0),
            head.prototype_velocities[1].unsqueeze(0)
        ).item()

        # Trajectory divergence: distance at t=1 minus distance at t=0
        trajectory_divergence = (inter_class_dist[-1] - inter_class_dist[0]).item()

        # Determine divergence pattern
        if trajectory_divergence > 0.05:
            divergence_pattern = "increasing"
        elif trajectory_divergence < -0.05:
            divergence_pattern = "decreasing"
        else:
            divergence_pattern = "stable"

        results[f'head_{head_idx}'] = {
            'hc_trajectory': hc_trajectory.numpy().tolist(),
            'mg_trajectory': mg_trajectory.numpy().tolist(),
            'inter_class_distance_over_time': inter_class_dist.numpy().tolist(),
            'hc_velocity_norm': hc_velocity_norm,
            'mg_velocity_norm': mg_velocity_norm,
            'origin_separation': origin_separation,
            'velocity_cosine_sim': velocity_cosine,
            'trajectory_divergence': trajectory_divergence,
            'divergence_pattern': divergence_pattern,
            'hc_origin': head.prototype_origins[0].detach().cpu().numpy().tolist(),
            'mg_origin': head.prototype_origins[1].detach().cpu().numpy().tolist(),
            'hc_velocity': head.prototype_velocities[0].detach().cpu().numpy().tolist(),
            'mg_velocity': head.prototype_velocities[1].detach().cpu().numpy().tolist(),
        }

        all_hc_velocities.append(hc_velocity_norm)
        all_mg_velocities.append(mg_velocity_norm)
        all_origin_separations.append(origin_separation)
        all_velocity_cosines.append(velocity_cosine)
        all_trajectory_divergences.append(trajectory_divergence)

    # Aggregate metrics
    all_velocity_norms = all_hc_velocities + all_mg_velocities

    results['aggregate'] = {
        'mean_hc_velocity_norm': float(np.mean(all_hc_velocities)),
        'mean_mg_velocity_norm': float(np.mean(all_mg_velocities)),
        'mean_velocity_norm': float(np.mean(all_velocity_norms)),
        'std_velocity_norm': float(np.std(all_velocity_norms)),
        'min_velocity_norm': float(np.min(all_velocity_norms)),
        'mean_origin_separation': float(np.mean(all_origin_separations)),
        'std_origin_separation': float(np.std(all_origin_separations)),
        'mean_velocity_cosine_sim': float(np.mean(all_velocity_cosines)),
        'mean_trajectory_divergence': float(np.mean(all_trajectory_divergences)),
        'all_heads_have_motion': bool(np.min(all_velocity_norms) > 0.05),
        'n_heads_with_positive_divergence': int(sum(d > 0 for d in all_trajectory_divergences)),
        't_values': t_values.numpy().tolist(),
    }

    return results


def compute_trajectory_pca(model, n_points: int = 20) -> Dict[str, Any]:
    """
    Compute PCA projection of trajectory prototypes for visualization.

    Returns:
        Dict with 2D PCA coordinates for trajectories
    """
    t_values = torch.linspace(0, 1, n_points)

    # Collect all trajectory points
    all_points = []
    point_labels = []  # (head_idx, class, time_idx)

    for head_idx, head in enumerate(model.heads):
        for t_idx, t in enumerate(t_values):
            t_tensor = torch.tensor([t], device=head.prototype_origins.device)
            pos = head.prototype_origins + t_tensor.unsqueeze(1) * head.prototype_velocities
            pos = pos.detach().cpu()

            # HC point
            all_points.append(pos[0].numpy())
            point_labels.append((head_idx, 0, t_idx))

            # MG point
            all_points.append(pos[1].numpy())
            point_labels.append((head_idx, 1, t_idx))

    all_points = np.array(all_points)

    # Fit PCA
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(all_points)

    # Organize by head and class
    result = {
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'trajectories_2d': {},
    }

    for head_idx in range(model.n_heads):
        result['trajectories_2d'][f'head_{head_idx}'] = {
            'hc': [],
            'mg': [],
        }

    for i, (head_idx, class_idx, t_idx) in enumerate(point_labels):
        class_name = 'hc' if class_idx == 0 else 'mg'
        result['trajectories_2d'][f'head_{head_idx}'][class_name].append(points_2d[i].tolist())

    # Compute centroids
    hc_points = points_2d[[i for i, (_, c, _) in enumerate(point_labels) if c == 0]]
    mg_points = points_2d[[i for i, (_, c, _) in enumerate(point_labels) if c == 1]]

    result['hc_centroid'] = hc_points.mean(axis=0).tolist()
    result['mg_centroid'] = mg_points.mean(axis=0).tolist()
    result['class_separation_in_pca'] = float(np.linalg.norm(
        np.array(result['hc_centroid']) - np.array(result['mg_centroid'])
    ))

    return result


def analyze_trajectory_prototypes(model, device: torch.device) -> Dict[str, Any]:
    """
    Complete trajectory prototype analysis.

    Args:
        model: Trained MHTPN model
        device: Torch device

    Returns:
        Dict with all trajectory analysis results
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # Compute trajectory evolution
        evolution_results = compute_trajectory_evolution(model, n_points=20)

        # Compute PCA for visualization
        pca_results = compute_trajectory_pca(model, n_points=20)

    # Combine results
    results = {
        'trajectory_evolution': evolution_results,
        'trajectory_pca': pca_results,
        'summary': {
            'mean_velocity_norm': evolution_results['aggregate']['mean_velocity_norm'],
            'all_heads_have_motion': evolution_results['aggregate']['all_heads_have_motion'],
            'mean_origin_separation': evolution_results['aggregate']['mean_origin_separation'],
            'mean_velocity_cosine_sim': evolution_results['aggregate']['mean_velocity_cosine_sim'],
            'mean_trajectory_divergence': evolution_results['aggregate']['mean_trajectory_divergence'],
            'pca_explained_variance': pca_results['pca_explained_variance'],
            'class_separation_in_pca': pca_results['class_separation_in_pca'],
        },
        'success_criteria': {
            'all_heads_have_motion': {
                'threshold': 'all velocity norms > 0.05',
                'value': evolution_results['aggregate']['min_velocity_norm'],
                'passed': evolution_results['aggregate']['all_heads_have_motion'],
            },
            'origin_separation_exists': {
                'threshold': 'mean separation > 0.5',
                'value': evolution_results['aggregate']['mean_origin_separation'],
                'passed': evolution_results['aggregate']['mean_origin_separation'] > 0.5,
            },
            'trajectories_not_parallel': {
                'threshold': 'mean cosine similarity < 0.95',
                'value': evolution_results['aggregate']['mean_velocity_cosine_sim'],
                'passed': abs(evolution_results['aggregate']['mean_velocity_cosine_sim']) < 0.95,
            },
        },
    }

    return results
