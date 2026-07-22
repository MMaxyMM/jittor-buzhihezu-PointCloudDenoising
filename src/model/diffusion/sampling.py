"""Deterministic samplers for conditional residual diffusion."""

from typing import Optional

import jittor as jt
import numpy as np


@jt.no_grad()
def ddim_sample(
    model,
    condition,
    observation_std,
    schedule,
    num_inference_steps: int,
    seed: int = 123,
    initial_noise: Optional[jt.Var] = None,
    clip_normalized_residual: Optional[float] = 8.0,
):
    """Sample normalized residuals with deterministic DDIM (eta=0)."""
    batch_size, num_points, dims = condition.shape
    if initial_noise is None:
        random_state = np.random.RandomState(seed)
        initial_noise = jt.array(
            random_state.normal(size=(batch_size, num_points, dims)).astype(np.float32)
        )
    state = initial_noise
    indices = schedule.inference_indices(num_inference_steps)
    clean = state

    for current_index, next_index in zip(indices[:-1], indices[1:]):
        current = np.full((batch_size,), current_index, dtype=np.int64)
        following = np.full((batch_size,), next_index, dtype=np.int64)
        alpha, sigma = schedule.coefficients(current)
        next_alpha, next_sigma = schedule.coefficients(following)
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
