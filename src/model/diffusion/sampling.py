"""Deterministic samplers for conditional residual diffusion."""

import math
from typing import Optional

import jittor as jt


def deterministic_point_noise(point_ids, channels: int = 3, seed: int = 123):
    """Device-only deterministic Gaussian noise keyed by global point ids."""
    ids = point_ids.float32()
    components = []
    for channel in range(channels):
        offset = float(seed * 0.1031 + channel * 17.371)
        u1 = jt.sin(ids * 12.9898 + offset) * 43758.5453
        u1 = u1 - jt.floor(u1)
        u1 = jt.maximum(jt.minimum(u1, 1.0 - 1e-6), 1e-6)
        u2 = jt.sin(ids * 78.233 + offset * 1.731) * 12345.6789
        u2 = u2 - jt.floor(u2)
        components.append(
            jt.sqrt(-2.0 * jt.log(u1)) * jt.cos(2.0 * math.pi * u2)
        )
    return jt.stack(components, dim=-1)


@jt.no_grad()
def ddim_sample(
    model,
    condition,
    observation_std,
    schedule,
    num_inference_steps: int,
    seed: int = 123,
    initial_noise: Optional[jt.Var] = None,
    point_ids: Optional[jt.Var] = None,
    clip_normalized_residual: Optional[float] = 8.0,
):
    """Sample normalized residuals with deterministic DDIM (eta=0)."""
    batch_size, num_points, dims = condition.shape
    if initial_noise is None:
        if point_ids is not None:
            initial_noise = deterministic_point_noise(
                point_ids, channels=dims, seed=seed
            )
        else:
            point_ids = jt.arange(num_points).reshape(1, num_points).broadcast(
                (batch_size, num_points)
            )
            initial_noise = deterministic_point_noise(
                point_ids, channels=dims, seed=seed
            )
    state = initial_noise
    indices = schedule.inference_indices(num_inference_steps)
    clean = state
    self_condition = None

    for current_index, next_index in zip(indices[:-1], indices[1:]):
        current = jt.ones((batch_size,)).int32() * int(current_index)
        following = jt.ones((batch_size,)).int32() * int(next_index)
        alpha, sigma = schedule.coefficients(current)
        next_alpha, next_sigma = schedule.coefficients(following)
        if getattr(model, "supports_self_conditioning", False):
            clean, predicted_noise = model.predict_clean_and_noise(
                state,
                condition,
                current,
                observation_std,
                alpha,
                sigma,
                self_condition=self_condition,
            )
            self_condition = clean.detach()
        else:
            clean, predicted_noise = model.predict_clean_and_noise(
                state,
                condition,
                current,
                observation_std,
                alpha,
                sigma,
            )
        if clip_normalized_residual is not None:
            limit = float(clip_normalized_residual)
            clean = jt.clamp(clean, -limit, limit)
        state = next_alpha * clean + next_sigma * predicted_noise

    return clean
