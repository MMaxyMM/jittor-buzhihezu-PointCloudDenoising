"""Shared patch extraction and fusion for full point-cloud inference."""

from math import ceil
from typing import Optional

import jittor as jt
import numpy as np


def farthest_point_sampling(pcls, num_points):
    """Sample FPS seeds with one device sync instead of one per seed."""
    points = np.asarray(pcls.numpy(), dtype=np.float32)
    batch_size, num_input_points, _ = points.shape
    num_points = min(int(num_points), int(num_input_points))
    selected = np.empty((batch_size, num_points), dtype=np.int32)
    for batch_idx in range(batch_size):
        distances = np.full(num_input_points, np.inf, dtype=np.float32)
        farthest = 0
        for sample_idx in range(num_points):
            selected[batch_idx, sample_idx] = farthest
            delta = points[batch_idx] - points[batch_idx, farthest]
            distances = np.minimum(
                distances, np.einsum("ij,ij->i", delta, delta)
            )
            farthest = int(np.argmax(distances))
    indices = jt.array(selected).int32()
    sampled = jt.stack(
        [pcls[batch_idx][indices[batch_idx]]
         for batch_idx in range(batch_size)],
        dim=0,
    )
    return sampled, indices


def knn_points(query, reference, k):
    """Return squared distances, indices, and neighbors for batched KNN."""
    distances = ((query.unsqueeze(2) - reference.unsqueeze(1)) ** 2).sum(-1)
    nearest_distances, indices = jt.topk(
        distances, k=k, dim=-1, largest=False
    )
    batch_size, num_reference, channels = reference.shape
    base = (jt.arange(batch_size) * num_reference).reshape(
        batch_size, 1, 1
    )
    flat_indices = (indices + base).reshape(-1)
    neighbors = reference.reshape(-1, channels)[flat_indices].reshape(
        batch_size, query.shape[1], k, channels
    )
    return nearest_distances, indices, neighbors


def patch_based_denoise(
    model,
    pcl_noisy,
    patch_size: int = 1000,
    seed_k: int = 6,
    seed_k_alpha: float = 1,
    inner_steps: Optional[int] = None,
    patch_batch_size: Optional[int] = None,
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

    if patch_batch_size is None:
        patch_batch_size = max(
            1, int(ceil(num_points / (float(seed_k_alpha) * patch_size)))
        )
    else:
        patch_batch_size = max(1, int(patch_batch_size))
    denoised_patches = []
    anchor_patches = []
    correction_patches = []
    for start in range(0, num_patches, patch_batch_size):
        current = patches[start:start + patch_batch_size]
        try:
            denoise_kwargs = {}
            if inner_steps is None:
                pass
            else:
                denoise_kwargs["num_steps"] = inner_steps
            if getattr(model, "accepts_point_ids", False):
                denoise_kwargs["point_ids"] = point_indices[
                    start:start + patch_batch_size
                ]
            output, auxiliary = model.denoise_langevin_dynamics(
                current, **denoise_kwargs
            )
        except Exception as exc:
            print("Denoise error:", exc)
            return None
        denoised_patches.append(output)
        if getattr(model, "returns_anchor_correction", False):
            if not isinstance(auxiliary, dict):
                raise RuntimeError(
                    "anchored patch model must return auxiliary predictions"
                )
            anchor_patches.append(auxiliary["anchor"])
            correction_patches.append(
                auxiliary["normalized_correction"]
            )

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
    if getattr(model, "returns_anchor_correction", False):
        anchors = jt.concat(anchor_patches, dim=0) + seed_expanded
        corrections = jt.concat(correction_patches, dim=0)
        anchor_sum = jt.zeros((num_points, 3)).scatter_(
            0,
            flat_indices.unsqueeze(1).broadcast((num_flat, 3)),
            anchors.reshape(-1, 3) * flat_weights,
            reduce="add",
        )
        correction_sum = jt.zeros((num_points, 3)).scatter_(
            0,
            flat_indices.unsqueeze(1).broadcast((num_flat, 3)),
            corrections.reshape(-1, 3) * flat_weights,
            reduce="add",
        )
        correction_square_sum = jt.zeros((num_points, 3)).scatter_(
            0,
            flat_indices.unsqueeze(1).broadcast((num_flat, 3)),
            corrections.reshape(-1, 3) ** 2 * flat_weights,
            reduce="add",
        )
        fused_anchor = anchor_sum / (weight_sum + 1e-12)
        fused_correction = correction_sum / (weight_sum + 1e-12)
        variance = jt.maximum(
            correction_square_sum / (weight_sum + 1e-12)
            - fused_correction ** 2,
            0.0,
        ).mean(dim=1, keepdims=True)
        beta = float(getattr(model, "blend_beta", 1.0))
        if getattr(model, "use_variance_gate", False):
            scale = float(getattr(model, "variance_gate_scale", 0.25))
            gate = beta * jt.exp(-variance / max(scale * scale, 1e-8))
            gate = jt.clamp(gate, 0.0, beta)
        else:
            gate = jt.ones((num_points, 1)) * beta
        refined = fused_anchor + (
            gate
            * float(getattr(model, "_current_inference_std", 1.0))
            * fused_correction
        )
        model._last_patch_diagnostics = {
            "fused_anchor": fused_anchor,
            "refined": refined,
            "normalized_correction": fused_correction,
            "correction_variance": variance,
            "gate": gate,
        }
        return jt.where(covered, refined, original)
    fused = prediction_sum / (weight_sum + 1e-12)
    return jt.where(covered, fused, original)
