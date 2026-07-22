"""Shared patch extraction and fusion for full point-cloud inference."""

from math import ceil
from typing import Optional

import jittor as jt


def farthest_point_sampling(pcls, num_points):
    """Sample seed points from batched point clouds using FPS."""
    batch_size, num_input_points, _ = pcls.shape
    sampled = []
    indices = []
    for batch_idx in range(batch_size):
        points = pcls[batch_idx]
        selected = []
        distances = jt.ones((num_input_points,)) * 1e10
        farthest = 0
        for _ in range(num_points):
            selected.append(farthest)
            centroid = points[farthest]
            current_distances = ((points - centroid) ** 2).sum(dim=1)
            distances = jt.minimum(distances, current_distances)
            farthest, _ = jt.argmax(distances, dim=-1)
            farthest = farthest.item()
        index = jt.array(selected).int32()
        sampled.append(points[index][None, ...])
        indices.append(index[None, ...])
    return jt.concat(sampled, dim=0), jt.concat(indices, dim=0)


def knn_points(query, reference, k):
    """Return squared distances, indices, and neighbors for batched KNN."""
    distances = ((query.unsqueeze(2) - reference.unsqueeze(1)) ** 2).sum(-1)
    nearest_distances, indices = jt.topk(
        distances, k=k, dim=-1, largest=False
    )
    neighbors = [
        reference[batch_idx][indices[batch_idx]]
        for batch_idx in range(query.shape[0])
    ]
    return nearest_distances, indices, jt.stack(neighbors, dim=0)


def patch_based_denoise(
    model,
    pcl_noisy,
    patch_size: int = 1000,
    seed_k: int = 6,
    seed_k_alpha: int = 1,
    inner_steps: Optional[int] = None,
):
    """Denoise a full cloud by overlapping patches and weighted fusion.

    The model must implement ``denoise_langevin_dynamics``. If ``inner_steps``
    is provided it is forwarded as ``num_steps``; legacy models keep their
    existing default when it is omitted.
    """
    if len(pcl_noisy.shape) != 2:
        raise ValueError(f"expected point cloud with shape (N, 3), got {pcl_noisy.shape}")

    num_points = pcl_noisy.shape[0]
    if num_points < patch_size:
        patch_size = num_points
    num_patches = max(1, int(seed_k * num_points / patch_size))
    pcl_batched = pcl_noisy.unsqueeze(0)

    seed_points, _ = farthest_point_sampling(pcl_batched, num_patches)
    patch_distances, point_indices, patches = knn_points(
        seed_points, pcl_batched, patch_size
    )

    patches = patches[0]
    patch_distances = patch_distances[0]
    point_indices = point_indices[0]
    seed_expanded = seed_points.squeeze(0).unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expanded

    patch_distances = patch_distances / (
        patch_distances[:, -1:].broadcast(patch_distances.shape) + 1e-8
    )

    patch_batch_size = max(1, int(ceil(num_points / (seed_k_alpha * patch_size))))
    denoised_patches = []
    for start in range(0, num_patches, patch_batch_size):
        current = patches[start:start + patch_batch_size]
        try:
            if inner_steps is None:
                output, _ = model.denoise_langevin_dynamics(current)
            else:
                output, _ = model.denoise_langevin_dynamics(
                    current, num_steps=inner_steps
                )
        except Exception as exc:
            print("Denoise error:", exc)
            return None
        denoised_patches.append(output)

    denoised_patches = jt.concat(denoised_patches, dim=0) + seed_expanded
    original = pcl_batched.squeeze(0)

    flat_indices = point_indices.reshape(-1)
    flat_weights = jt.exp(-patch_distances).reshape(-1, 1)
    weighted_predictions = denoised_patches.reshape(-1, 3) * flat_weights
    num_flat = flat_indices.shape[0]

    prediction_sum = jt.zeros((num_points, 3)).scatter_(
        0,
        flat_indices.unsqueeze(1).broadcast((num_flat, 3)),
        weighted_predictions,
        reduce="add",
    )
    weight_sum = jt.zeros((num_points, 1)).scatter_(
        0,
        flat_indices.unsqueeze(1).broadcast((num_flat, 1)),
        flat_weights,
        reduce="add",
    )
    covered = (weight_sum > 1e-12).broadcast((num_points, 3))
    fused = prediction_sum / (weight_sum + 1e-12)
    return jt.where(covered, fused, original)
