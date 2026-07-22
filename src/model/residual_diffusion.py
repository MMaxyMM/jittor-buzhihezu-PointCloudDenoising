"""Conditional residual diffusion for point-cloud denoising."""

import math
from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from ..data.asset import Asset
from .diffusion.sampling import ddim_sample
from .diffusion.schedule import (
    CosineSchedule,
    q_sample,
    v_target,
    v_to_clean_and_noise,
)
from .feature import Decoder, DynamicEdgeConv, get_knn_idx
from .patch_inference import patch_based_denoise
from .point_transformer import LocalPointTransformerEncoder
from .spec import ModelSpec


def _sinusoidal_embedding(values, embedding_dim: int):
    half_dim = embedding_dim // 2
    if half_dim < 1:
        return values.reshape(-1, 1)
    denominator = max(half_dim - 1, 1)
    frequencies = jt.exp(
        jt.arange(half_dim).float32()
        * (-math.log(10000.0) / denominator)
    )
    arguments = values.reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = jt.concat([jt.sin(arguments), jt.cos(arguments)], dim=-1)
    if embedding_dim % 2:
        embedding = jt.concat(
            [embedding, jt.zeros((embedding.shape[0], 1))], dim=-1
        )
    return embedding


class DiffusionConditionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    def execute(self, time_value, log_noise_std):
        time_embedding = _sinusoidal_embedding(
            time_value * 1000.0, self.embedding_dim
        )
        scale_embedding = _sinusoidal_embedding(
            log_noise_std, self.embedding_dim
        )
        return self.mlp(jt.concat([time_embedding, scale_embedding], dim=-1))


class ConditionalDGCNN(nn.Module):
    """DGCNN-like encoder whose KNN graph is defined only by observed XYZ."""

    def __init__(
        self,
        k: int,
        embedding_dim: int,
        condition_dim: int,
        input_dim: int = 6,
    ):
        super().__init__()
        self.k = k
        first_dim = embedding_dim // 8
        second_dim = embedding_dim // 4
        self.conv1 = DynamicEdgeConv(input_dim, first_dim)
        self.conv2 = DynamicEdgeConv(first_dim, second_dim)
        self.conv3 = DynamicEdgeConv(
            first_dim + second_dim, embedding_dim, activation=None
        )
        self.film1 = nn.Linear(condition_dim, first_dim * 2)
        self.film2 = nn.Linear(condition_dim, second_dim * 2)
        self.film3 = nn.Linear(condition_dim, embedding_dim * 2)
        self.embedding_dim = embedding_dim

    def _edge_index(self, geometry):
        batch_size, num_points, _ = geometry.shape
        indices = get_knn_idx(geometry, geometry, self.k + 1)[:, :, 1:]
        base = (jt.arange(batch_size) * num_points).reshape(
            batch_size, 1, 1
        )
        source = (indices + base).reshape(-1)
        destination = jt.arange(num_points).reshape(
            1, num_points, 1
        ).broadcast((batch_size, num_points, self.k))
        destination = (destination + base).reshape(-1)
        return jt.stack([source, destination], dim=0)

    @staticmethod
    def _apply_film(features, condition, projection):
        batch_size, _, channels = features.shape
        parameters = projection(condition)
        gamma = parameters[:, :channels].reshape(batch_size, 1, channels)
        beta = parameters[:, channels:].reshape(batch_size, 1, channels)
        return features * (1.0 + gamma) + beta

    def execute(self, features, geometry, condition):
        batch_size, num_points, _ = features.shape
        edge_index = self._edge_index(geometry)

        first = self.conv1(features.reshape(batch_size * num_points, -1), edge_index)
        first = first.reshape(batch_size, num_points, -1)
        first = self._apply_film(first, condition, self.film1)

        second = self.conv2(first.reshape(batch_size * num_points, -1), edge_index)
        second = second.reshape(batch_size, num_points, -1)
        second = self._apply_film(second, condition, self.film2)

        combined = jt.concat([first, second], dim=-1)
        third = self.conv3(combined.reshape(batch_size * num_points, -1), edge_index)
        third = third.reshape(batch_size, num_points, -1)
        return self._apply_film(third, condition, self.film3)


def estimate_noise_std(points: np.ndarray, k: int = 16, max_points: int = 8192):
    """Robust local-PCA estimate used only when explicitly enabled at inference."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < k + 1:
        return 0.0
    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        query_points = points[indices]
    else:
        query_points = points
    tree = cKDTree(points)
    _, neighbor_indices = tree.query(query_points, k=k + 1)
    neighbors = points[neighbor_indices[:, 1:]]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / k
    smallest_eigenvalue = np.linalg.eigvalsh(covariance)[:, 0]
    return float(np.sqrt(np.median(np.maximum(smallest_eigenvalue, 0.0))))


class ResidualDiffusionModule(ModelSpec):
    """Observation-conditioned diffusion over normalized clean-minus-noisy residuals."""

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.frame_knn = int(cfg.get("frame_knn", 16))
        self.num_train_points = int(cfg.get("num_train_points", 128))
        self.patch_size = int(cfg.get("patch_size", 1000))
        self.seed_k = int(cfg.get("seed_k", 6))
        self.seed_k_alpha = int(cfg.get("seed_k_alpha", 1))
        self.num_inference_steps = int(cfg.get("num_inference_steps", 4))
        self.predict_rounds = int(cfg.get("predict_rounds", 1))
        self.sampling_seed = int(cfg.get("sampling_seed", 123))
        self.lambda_l1 = float(cfg.get("lambda_l1", 0.1))
        self.lambda_cd = float(cfg.get("lambda_cd", 0.0))
        self.objective = cfg.get("objective", "diffusion")
        self.prediction_type = cfg.get("prediction_type", "v")
        self.condition_on_time = bool(cfg.get("condition_on_time", True))
        self.condition_on_observation_std = bool(
            cfg.get("condition_on_observation_std", True)
        )
        self.noise_std_min = float(cfg.get("noise_std_min", 0.005))
        self.noise_std_max = float(cfg.get("noise_std_max", 0.020))
        self.inference_noise_mode = cfg.get("inference_noise_mode", "fixed")
        self.inference_noise_std = float(cfg.get("inference_noise_std", 0.0125))
        self.clip_normalized_residual = cfg.get(
            "clip_normalized_residual", 8.0
        )

        condition_dim = int(cfg.get("condition_embedding_dim", 64))
        feature_dim = int(cfg.get("feat_embedding_dim", 256))
        self.backbone = cfg.get("backbone", "dgcnn")
        self.condition_embedding = DiffusionConditionEmbedding(condition_dim)
        if self.backbone == "dgcnn":
            self.encoder = ConditionalDGCNN(
                k=self.frame_knn,
                embedding_dim=feature_dim,
                condition_dim=condition_dim,
            )
        elif self.backbone == "point_transformer":
            self.encoder = LocalPointTransformerEncoder(
                k=self.frame_knn,
                embedding_dim=feature_dim,
                condition_dim=condition_dim,
                num_blocks=int(cfg.get("transformer_blocks", 3)),
            )
        else:
            raise ValueError(f"unsupported backbone: {self.backbone}")
        self.decoder = Decoder(
            z_dim=feature_dim,
            dim=3,
            out_dim=3,
            hidden_size=int(cfg.get("decoder_hidden_dim", 64)),
        )
        self.schedule = CosineSchedule(
            num_train_steps=int(cfg.get("num_train_steps", 1000)),
            cosine_s=float(cfg.get("cosine_s", 0.008)),
        )
        self._current_inference_std = self.inference_noise_std

    def _condition(self, timestep, observation_std):
        if not isinstance(timestep, jt.Var):
            timestep = jt.array(np.asarray(timestep, dtype=np.float32))
        timestep = timestep.float32().reshape(-1)
        time_value = timestep / float(self.schedule.num_train_steps)
        if not self.condition_on_time:
            time_value = jt.zeros_like(time_value)

        observation_std = observation_std.reshape(-1)
        safe_std = jt.maximum(observation_std, jt.array(1e-6))
        log_std = jt.log(safe_std)
        if not self.condition_on_observation_std:
            log_std = jt.zeros_like(log_std)
        return self.condition_embedding(time_value, log_std)

    def _network(self, state, noisy_condition, timestep, observation_std):
        condition_embedding = self._condition(timestep, observation_std)
        features = jt.concat([noisy_condition, state], dim=-1)
        encoded = self.encoder(features, noisy_condition, condition_embedding)
        batch_size, num_points, feature_dim = encoded.shape
        return self.decoder(
            c=encoded.reshape(-1, feature_dim)
        ).reshape(batch_size, num_points, 3)

    def predict_clean_and_noise(
        self,
        state,
        noisy_condition,
        timestep,
        observation_std,
        alpha,
        sigma,
    ):
        prediction = self._network(
            state, noisy_condition, timestep, observation_std
        )
        if self.prediction_type == "v":
            return v_to_clean_and_noise(
                state, prediction, alpha, sigma
            )
        if self.prediction_type == "epsilon":
            predicted_noise = prediction
            clean = (state - sigma * predicted_noise) / (alpha + 1e-5)
            return clean, predicted_noise
        raise ValueError(f"unsupported prediction_type: {self.prediction_type}")

    def get_supervised_loss(
        self, pc_noisy, pc_clean, observation_std
    ):
        batch_size, num_points, _ = pc_noisy.shape
        observation_std = observation_std.reshape(batch_size, 1, 1)
        normalized_clean = (pc_clean - pc_noisy) / (
            observation_std + 1e-6
        )

        if self.objective == "direct":
            timestep = np.zeros((batch_size,), dtype=np.int64)
            state = jt.zeros_like(normalized_clean)
            prediction = self._network(
                state, pc_noisy, timestep, observation_std
            )
            return jt.abs(prediction - normalized_clean).mean()
        if self.objective != "diffusion":
            raise ValueError(f"unsupported objective: {self.objective}")

        timestep = np.random.randint(
            1,
            self.schedule.num_train_steps + 1,
            size=(batch_size,),
            dtype=np.int64,
        )
        alpha, sigma = self.schedule.coefficients(timestep)
        noise = jt.array(
            np.random.normal(size=normalized_clean.shape).astype(np.float32)
        )
        state = q_sample(normalized_clean, noise, alpha, sigma)
        prediction = self._network(
            state, pc_noisy, timestep, observation_std
        )
        target = (
            v_target(normalized_clean, noise, alpha, sigma)
            if self.prediction_type == "v"
            else noise
        )

        point_indices = np.random.permutation(num_points)[
            :min(self.num_train_points, num_points)
        ]
        prediction_sample = prediction[:, point_indices, :]
        target_sample = target[:, point_indices, :]
        diffusion_loss = ((prediction_sample - target_sample) ** 2).mean()

        if self.lambda_l1 <= 0 and self.lambda_cd <= 0:
            return diffusion_loss
        if self.prediction_type == "v":
            predicted_clean, _ = v_to_clean_and_noise(
                state, prediction, alpha, sigma
            )
        else:
            predicted_clean = (state - sigma * prediction) / (alpha + 1e-5)
        predicted_clean_sample = predicted_clean[:, point_indices, :]
        normalized_clean_sample = normalized_clean[:, point_indices, :]
        total_loss = diffusion_loss
        if self.lambda_l1 > 0:
            reconstruction_l1 = jt.abs(
                predicted_clean_sample - normalized_clean_sample
            ).mean()
            total_loss = total_loss + self.lambda_l1 * reconstruction_l1
        if self.lambda_cd > 0:
            predicted_points = (
                pc_noisy[:, point_indices, :]
                + predicted_clean_sample * observation_std
            )
            clean_points = pc_clean[:, point_indices, :]
            pairwise_squared = (
                (
                    predicted_points.unsqueeze(2)
                    - clean_points.unsqueeze(1)
                )
                ** 2
            ).sum(dim=-1)
            symmetric_cd = (
                pairwise_squared.min(dim=2).mean(dim=1)
                + pairwise_squared.min(dim=1).mean(dim=1)
            )
            normalized_cd = (
                symmetric_cd
                / (observation_std.reshape(batch_size) ** 2 + 1e-8)
            ).mean()
            total_loss = total_loss + self.lambda_cd * normalized_cd
        return total_loss

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        noise_std = batch["noise_std"].reshape(-1, 1)
        return {
            "loss": self.get_supervised_loss(
                pc_noisy, pc_clean, noise_std
            )
        }

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    @jt.no_grad()
    def denoise_langevin_dynamics(self, pcl_noisy, num_steps=None):
        batch_size = pcl_noisy.shape[0]
        observation_std = jt.ones((batch_size, 1, 1)) * float(
            self._current_inference_std
        )
        if self.objective == "direct":
            timestep = np.zeros((batch_size,), dtype=np.int64)
            state = jt.zeros_like(pcl_noisy)
            normalized_residual = self._network(
                state, pcl_noisy, timestep, observation_std
            )
        else:
            normalized_residual = ddim_sample(
                model=self,
                condition=pcl_noisy,
                observation_std=observation_std,
                schedule=self.schedule,
                num_inference_steps=(
                    self.num_inference_steps
                    if num_steps is None else int(num_steps)
                ),
                seed=self.sampling_seed,
                clip_normalized_residual=self.clip_normalized_residual,
            )
        denoised = pcl_noisy + normalized_residual * observation_std
        return denoised, None

    def _inference_std_for_cloud(self, points):
        if self.inference_noise_mode == "fixed":
            return self.inference_noise_std
        if self.inference_noise_mode == "estimate":
            estimated = estimate_noise_std(points, k=self.frame_knn)
            return float(
                np.clip(estimated, self.noise_std_min, self.noise_std_max)
            )
        raise ValueError(
            f"unsupported inference_noise_mode: {self.inference_noise_mode}"
        )

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        results = []
        for pc_noisy in batch["pc_noisy"]:
            original_numpy = pc_noisy.detach().numpy()
            self._current_inference_std = self._inference_std_for_cloud(
                original_numpy
            )
            current = pc_noisy
            for _ in range(self.predict_rounds):
                denoised = patch_based_denoise(
                    self,
                    current,
                    patch_size=self.patch_size,
                    seed_k=self.seed_k,
                    seed_k_alpha=self.seed_k_alpha,
                    inner_steps=(
                        1 if self.objective == "direct"
                        else self.num_inference_steps
                    ),
                )
                if denoised is None:
                    break
                current = denoised
            output = current.detach().numpy().astype(np.float32, copy=False)
            if output.shape != original_numpy.shape:
                raise RuntimeError(
                    f"denoised shape {output.shape} does not match input "
                    f"{original_numpy.shape}"
                )
            results.append({"pc_denoised": output})
        return results

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        results = []
        for asset in batch:
            if self.is_predict():
                results.append({"pc_noisy": asset.sampled_vertices_noisy})
                continue
            if asset.meta is None:
                raise RuntimeError("diffusion training requires patch metadata")
            required = ("pc_noisy", "pc_clean", "noise_std")
            missing = [key for key in required if key not in asset.meta]
            if missing:
                raise RuntimeError(
                    f"diffusion patch metadata is missing: {missing}"
                )
            results.append({key: asset.meta[key] for key in required})
        return results
