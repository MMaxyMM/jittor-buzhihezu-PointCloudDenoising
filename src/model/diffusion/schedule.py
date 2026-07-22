"""Cosine variance schedule and v-parameterization helpers."""

import math
from typing import Tuple

import jittor as jt
import numpy as np


class CosineSchedule:
    """Continuous-endpoint cosine schedule stored at discrete indices [0, T]."""

    def __init__(self, num_train_steps: int = 1000, cosine_s: float = 0.008):
        if num_train_steps < 2:
            raise ValueError("num_train_steps must be at least 2")
        self.num_train_steps = int(num_train_steps)
        steps = np.arange(self.num_train_steps + 1, dtype=np.float64)
        phase = ((steps / self.num_train_steps) + cosine_s) / (1.0 + cosine_s)
        alpha_bar = np.cos(phase * math.pi / 2.0) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        alpha_bar = np.clip(alpha_bar, 0.0, 1.0)
        self.alphas_np = np.sqrt(alpha_bar).astype(np.float32)
        self.sigmas_np = np.sqrt(1.0 - alpha_bar).astype(np.float32)

    def coefficients(self, indices) -> Tuple[jt.Var, jt.Var]:
        """Return broadcastable alpha and sigma for integer timestep indices."""
        if isinstance(indices, jt.Var):
            index_array = indices.numpy().astype(np.int64)
        else:
            index_array = np.asarray(indices, dtype=np.int64)
        alpha = jt.array(self.alphas_np[index_array]).float32()
        sigma = jt.array(self.sigmas_np[index_array]).float32()
        return alpha.reshape(-1, 1, 1), sigma.reshape(-1, 1, 1)

    def inference_indices(self, num_inference_steps: int) -> np.ndarray:
        """Descending unique indices including both the noisy and clean ends."""
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        indices = np.rint(
            np.linspace(self.num_train_steps, 0, num_inference_steps + 1)
        ).astype(np.int64)
        return np.unique(indices)[::-1]


def q_sample(clean_residual, noise, alpha, sigma):
    return alpha * clean_residual + sigma * noise


def v_target(clean_residual, noise, alpha, sigma):
    return alpha * noise - sigma * clean_residual


def v_to_clean_and_noise(noisy_residual, velocity, alpha, sigma):
    """Recover x0 and epsilon from v-prediction."""
    clean = alpha * noisy_residual - sigma * velocity
    noise = sigma * noisy_residual + alpha * velocity
    return clean, noise
