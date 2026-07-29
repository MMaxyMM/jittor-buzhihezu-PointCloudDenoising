"""Frozen-direct-anchor Point Transformer diffusion refiner."""

import os
from typing import Dict, List, Optional

import jittor as jt
import numpy as np
from jittor import nn

from ..data.asset import Asset
from .diffusion.sampling import ddim_sample
from .diffusion.schedule import CosineSchedule, q_sample, v_target, v_to_clean_and_noise
from .patch_inference import patch_based_denoise
from .point_transformer import LocalPointTransformerEncoder
from .residual_diffusion import (
    DiffusionConditionEmbedding,
    ResidualDiffusionModule,
    estimate_noise_std,
)
from .feature import Decoder
from .spec import ModelSpec


class AnchoredResidualDiffusionModule(ModelSpec):
    """Diffuse only the residual error left by a frozen Direct DGCNN."""

    accepts_point_ids = True
    supports_self_conditioning = True
    returns_anchor_correction = True

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        anchor_checkpoint = cfg.get("anchor_ckpt")
        if not anchor_checkpoint or not os.path.isfile(anchor_checkpoint):
            raise FileNotFoundError(
                "anchor_ckpt must point to a trained Direct DGCNN checkpoint: "
                f"{anchor_checkpoint}"
            )

        anchor_config = dict(cfg.get("anchor_model", {}))
        anchor_config.pop("__target__", None)
        if not anchor_config:
            raise ValueError("anchor_model config is required to reconstruct the anchor")
        self.anchor = ResidualDiffusionModule(anchor_config, {})
        self.anchor.load(anchor_checkpoint)
        self.anchor.eval()
        for parameter in self.anchor.parameters():
            parameter.stop_grad()

        self.frame_knn = int(cfg.get("frame_knn", 24))
        self.patch_size = int(cfg.get("patch_size", 1000))
        self.seed_k = int(cfg.get("seed_k", 6))
        self.seed_k_alpha = float(cfg.get("seed_k_alpha", 1))
        self.inference_patch_batch_size = cfg.get("inference_patch_batch_size")
        if self.inference_patch_batch_size is not None:
            self.inference_patch_batch_size = int(self.inference_patch_batch_size)
        self.num_inference_steps = int(cfg.get("num_inference_steps", 8))
        self.predict_rounds = int(cfg.get("predict_rounds", 1))
        self.sampling_seed = int(cfg.get("sampling_seed", 123))
        configured_seeds = cfg.get("sampling_seeds")
        self.sampling_seeds = (
            [self.sampling_seed]
            if configured_seeds is None
            else [int(seed) for seed in configured_seeds]
        )
        if not 1 <= len(self.sampling_seeds) <= 2:
            raise ValueError("sampling_seeds must contain one or two seeds")
        self.inference_noise_std = float(cfg.get("inference_noise_std", 0.0125))
        self.inference_noise_mode = cfg.get("inference_noise_mode", "fixed")
        self.noise_std_min = float(cfg.get("noise_std_min", 0.005))
        self.noise_std_max = float(cfg.get("noise_std_max", 0.020))
        self.clip_normalized_residual = float(
            cfg.get("clip_normalized_residual", 3.0)
        )
        self.self_condition_probability = float(
            cfg.get("self_condition_probability", 0.5)
        )

        self.min_snr_gamma = float(cfg.get("min_snr_gamma", 5.0))
        self.lambda_smooth_l1 = float(cfg.get("lambda_smooth_l1", 0.5))
        self.smooth_l1_beta = float(cfg.get("smooth_l1_beta", 0.25))
        self.lambda_cd = float(cfg.get("lambda_cd", 0.0))
        self.lambda_point_to_plane = float(
            cfg.get("lambda_point_to_plane", 0.0)
        )
        self.cd_num_points = int(cfg.get("cd_num_points", 256))

        self.blend_beta = float(cfg.get("blend_beta", 0.75))
        self.use_variance_gate = bool(cfg.get("use_variance_gate", True))
        self.variance_gate_scale = float(cfg.get("variance_gate_scale", 0.25))

        condition_dim = int(cfg.get("condition_embedding_dim", 64))
        feature_dim = int(cfg.get("feat_embedding_dim", 192))
        self.condition_embedding = DiffusionConditionEmbedding(condition_dim)
        self.encoder = LocalPointTransformerEncoder(
            k=self.frame_knn,
            embedding_dim=feature_dim,
            condition_dim=condition_dim,
            input_dim=12,
            num_blocks=int(cfg.get("transformer_blocks", 4)),
            expansion=2,
        )
        self.decoder = Decoder(
            z_dim=feature_dim,
            dim=3,
            out_dim=3,
            hidden_size=int(cfg.get("decoder_hidden_dim", 96)),
        )
        self.schedule = CosineSchedule(
            num_train_steps=int(cfg.get("num_train_steps", 1000)),
            cosine_s=float(cfg.get("cosine_s", 0.008)),
        )
        self._current_inference_std = self.inference_noise_std
        self._sampling_anchor_residual = None

    def train(self):
        super().train()
        self.anchor.eval()
        return self

    def get_optim_parameters(self):
        """Exclude the frozen anchor from optimizer and EMA state."""
        modules = (
            self.condition_embedding,
            self.encoder,
            self.decoder,
        )
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]

    def _condition(self, timestep, observation_std):
        if not isinstance(timestep, jt.Var):
            timestep = jt.array(np.asarray(timestep, dtype=np.float32))
        time_value = timestep.float32().reshape(-1) / float(
            self.schedule.num_train_steps
        )
        log_std = jt.log(jt.maximum(observation_std.reshape(-1), 1e-6))
        return self.condition_embedding(time_value, log_std)

    @jt.no_grad()
    def _anchor_prediction(self, noisy, observation_std):
        batch_size = noisy.shape[0]
        timestep = jt.zeros((batch_size,)).int32()
        normalized_residual = self.anchor._network(
            jt.zeros_like(noisy), noisy, timestep, observation_std
        )
        anchor = noisy + normalized_residual * observation_std
        return anchor, normalized_residual

    def _network(
        self,
        state,
        noisy,
        anchor_residual,
        timestep,
        observation_std,
        self_condition=None,
    ):
        if self_condition is None:
            self_condition = jt.zeros_like(state)
        features = jt.concat(
            [noisy, anchor_residual, state, self_condition], dim=-1
        )
        encoded = self.encoder(
            features, noisy, self._condition(timestep, observation_std)
        )
        batch_size, num_points, feature_dim = encoded.shape
        return self.decoder(
            c=encoded.reshape(-1, feature_dim)
        ).reshape(batch_size, num_points, 3)

    def predict_clean_and_noise(
        self,
        state,
        condition,
        timestep,
        observation_std,
        alpha,
        sigma,
        self_condition=None,
    ):
        anchor_residual = self._sampling_anchor_residual
        if anchor_residual is None:
            _, anchor_residual = self._anchor_prediction(
                condition, observation_std
            )
        velocity = self._network(
            state,
            condition,
            anchor_residual,
            timestep,
            observation_std,
            self_condition=self_condition,
        )
        return v_to_clean_and_noise(state, velocity, alpha, sigma)

    @staticmethod
    def _smooth_l1(error, beta):
        absolute = jt.abs(error)
        return jt.where(
            absolute < beta,
            0.5 * absolute * absolute / beta,
            absolute - 0.5 * beta,
        )

    def get_supervised_loss(
        self,
        pc_noisy,
        pc_clean,
        observation_std,
        clean_normals: Optional[jt.Var] = None,
    ):
        batch_size, num_points, _ = pc_noisy.shape
        observation_std = observation_std.reshape(batch_size, 1, 1)
        anchor, anchor_residual = self._anchor_prediction(
            pc_noisy, observation_std
        )
        normalized_clean = (pc_clean - anchor) / (observation_std + 1e-6)

        timestep = np.random.randint(
            1, self.schedule.num_train_steps, size=(batch_size,), dtype=np.int32
        )
        alpha, sigma = self.schedule.coefficients(timestep)
        noise = jt.randn(normalized_clean.shape)
        state = q_sample(normalized_clean, noise, alpha, sigma)
        target = v_target(normalized_clean, noise, alpha, sigma)

        self_condition = None
        if np.random.rand() < self.self_condition_probability:
            with jt.no_grad():
                first_velocity = self._network(
                    state,
                    pc_noisy,
                    anchor_residual,
                    timestep,
                    observation_std,
                )
                self_condition, _ = v_to_clean_and_noise(
                    state, first_velocity, alpha, sigma
                )
                self_condition = self_condition.detach()
        velocity = self._network(
            state,
            pc_noisy,
            anchor_residual,
            timestep,
            observation_std,
            self_condition=self_condition,
        )

        per_sample_mse = (
            (velocity - target) ** 2
        ).mean(dim=2).mean(dim=1)
        snr = (alpha.reshape(-1) ** 2) / (
            sigma.reshape(-1) ** 2 + 1e-8
        )
        weights = jt.minimum(snr, self.min_snr_gamma) / (snr + 1.0)
        weights = weights / (weights.mean() + 1e-8)
        total = (per_sample_mse * weights).mean()

        predicted_clean, _ = v_to_clean_and_noise(
            state, velocity, alpha, sigma
        )
        if self.lambda_smooth_l1 > 0:
            total = total + self.lambda_smooth_l1 * self._smooth_l1(
                predicted_clean - normalized_clean, self.smooth_l1_beta
            ).mean()

        predicted_points = anchor + predicted_clean * observation_std
        if self.lambda_cd > 0:
            sample_size = min(self.cd_num_points, num_points)
            indices = np.random.permutation(num_points)[:sample_size]
            predicted_sample = predicted_points[:, indices, :]
            clean_sample = pc_clean[:, indices, :]
            squared = (
                predicted_sample.unsqueeze(2) - clean_sample.unsqueeze(1)
            ) ** 2
            squared = squared.sum(dim=-1)
            cd = squared.min(dim=2).mean(dim=1) + squared.min(dim=1).mean(dim=1)
            normalized_cd = (
                cd / (observation_std.reshape(batch_size) ** 2 + 1e-8)
            ).mean()
            total = total + self.lambda_cd * normalized_cd

        if self.lambda_point_to_plane > 0:
            if clean_normals is None:
                raise RuntimeError(
                    "point-to-plane loss requires clean_normals in the cache"
                )
            normal_error = (
                (predicted_points - pc_clean) * clean_normals
            ).sum(dim=-1)
            point_to_plane = (
                normal_error ** 2
                / (observation_std.reshape(batch_size, 1) ** 2 + 1e-8)
            ).mean()
            total = total + self.lambda_point_to_plane * point_to_plane
        return total

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        clean_normals = batch.get("clean_normals")
        if clean_normals is not None:
            clean_normals = clean_normals.reshape(-1, patch_size, 3)
        return {
            "loss": self.get_supervised_loss(
                batch["pc_noisy"].reshape(-1, patch_size, 3),
                batch["pc_clean"].reshape(-1, patch_size, 3),
                batch["noise_std"].reshape(-1, 1),
                clean_normals=clean_normals,
            )
        }

    def execute(self, **kwargs):
        return self.training_step(**kwargs)

    @jt.no_grad()
    def denoise_langevin_dynamics(
        self, pcl_noisy, num_steps=None, point_ids=None
    ):
        batch_size = pcl_noisy.shape[0]
        observation_std = jt.ones((batch_size, 1, 1)) * float(
            self._current_inference_std
        )
        anchor, anchor_residual = self._anchor_prediction(
            pcl_noisy, observation_std
        )
        if self.blend_beta == 0.0:
            return anchor, {
                "anchor": anchor,
                "normalized_correction": jt.zeros_like(anchor),
            }
        self._sampling_anchor_residual = anchor_residual
        try:
            corrections = [
                ddim_sample(
                    model=self,
                    condition=pcl_noisy,
                    observation_std=observation_std,
                    schedule=self.schedule,
                    num_inference_steps=(
                        self.num_inference_steps
                        if num_steps is None else int(num_steps)
                    ),
                    seed=seed,
                    point_ids=point_ids,
                    clip_normalized_residual=self.clip_normalized_residual,
                )
                for seed in self.sampling_seeds
            ]
            correction = sum(corrections[1:], corrections[0]) / len(
                corrections
            )
        finally:
            self._sampling_anchor_residual = None
        correction = jt.clamp(
            correction,
            -self.clip_normalized_residual,
            self.clip_normalized_residual,
        )
        refined = anchor + correction * observation_std
        return refined, {
            "anchor": anchor,
            "normalized_correction": correction,
        }

    def _inference_std_for_cloud(self, points):
        if self.inference_noise_mode == "fixed":
            return self.inference_noise_std
        if self.inference_noise_mode == "estimate":
            return float(np.clip(
                estimate_noise_std(points, k=self.frame_knn),
                self.noise_std_min,
                self.noise_std_max,
            ))
        raise ValueError(
            f"unsupported inference_noise_mode: {self.inference_noise_mode}"
        )

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        results = []
        for pc_noisy in batch["pc_noisy"]:
            original = pc_noisy.detach().numpy()
            self._current_inference_std = self._inference_std_for_cloud(original)
            denoised = patch_based_denoise(
                self,
                pc_noisy,
                patch_size=self.patch_size,
                seed_k=self.seed_k,
                seed_k_alpha=self.seed_k_alpha,
                inner_steps=self.num_inference_steps,
                patch_batch_size=self.inference_patch_batch_size,
            )
            if denoised is None:
                denoised = pc_noisy
            results.append({
                "pc_denoised": denoised.detach().numpy().astype(
                    np.float32, copy=False
                )
            })
        return results

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        results = []
        for asset in batch:
            if self.is_predict():
                results.append({"pc_noisy": asset.sampled_vertices_noisy})
                continue
            if asset.meta is None:
                raise RuntimeError("anchored diffusion requires patch metadata")
            required = ["pc_noisy", "pc_clean", "noise_std"]
            if self.lambda_point_to_plane > 0:
                required.append("clean_normals")
            missing = [key for key in required if key not in asset.meta]
            if missing:
                raise RuntimeError(f"patch metadata is missing: {missing}")
            results.append({key: asset.meta[key] for key in required})
        return results
