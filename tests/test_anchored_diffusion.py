import os
import tempfile
import unittest
from copy import deepcopy

import jittor as jt
import numpy as np

from src.model.anchored_residual_diffusion import AnchoredResidualDiffusionModule
from src.model.parse import get_model
from src.model.patch_inference import patch_based_denoise
from src.model.residual_diffusion import ResidualDiffusionModule


def anchor_config():
    return {
        "backbone": "dgcnn",
        "frame_knn": 3,
        "feat_embedding_dim": 32,
        "decoder_hidden_dim": 16,
        "condition_embedding_dim": 16,
        "objective": "direct",
        "condition_on_time": False,
        "num_train_steps": 20,
        "num_inference_steps": 1,
        "patch_size": 12,
        "seed_k": 1,
        "seed_k_alpha": 1,
        "inference_noise_std": 0.01,
    }


class AnchoredDiffusionTest(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 0
        np.random.seed(7)
        self.directory = tempfile.TemporaryDirectory()
        self.anchor_path = os.path.join(self.directory.name, "anchor.pkl")
        ResidualDiffusionModule(anchor_config(), {}).save(self.anchor_path)

    def tearDown(self):
        self.directory.cleanup()

    def config(self):
        return {
            "__target__": "AnchoredResidualDiffusionModule",
            "anchor_ckpt": self.anchor_path,
            "anchor_model": deepcopy(anchor_config()),
            "frame_knn": 3,
            "feat_embedding_dim": 32,
            "decoder_hidden_dim": 16,
            "condition_embedding_dim": 16,
            "transformer_blocks": 1,
            "num_train_steps": 20,
            "num_inference_steps": 2,
            "patch_size": 12,
            "seed_k": 1,
            "seed_k_alpha": 1,
            "cd_num_points": 6,
            "lambda_cd": 0.02,
            "lambda_point_to_plane": 0.15,
            "self_condition_probability": 1.0,
            "inference_noise_std": 0.01,
        }

    def batch(self):
        noisy = jt.array(
            (np.random.randn(1, 1, 12, 3) * 0.01).astype(np.float32)
        )
        return {
            "pc_noisy": noisy,
            "pc_clean": jt.zeros_like(noisy),
            "clean_normals": jt.array(
                np.tile(
                    np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    (1, 1, 12, 1),
                )
            ),
            "noise_std": jt.ones((1, 1, 1)) * 0.01,
        }

    def test_registry_freeze_self_condition_and_geometry_loss(self):
        model = get_model(self.config(), transform_config={})
        self.assertIsInstance(model, AnchoredResidualDiffusionModule)
        self.assertLess(
            len(model.get_optim_parameters()), len(list(model.parameters()))
        )
        loss = model.training_step(self.batch())["loss"]
        self.assertTrue(np.isfinite(float(loss.item())))

    def test_anchor_only_patch_regression_and_deterministic_sampling(self):
        model = get_model(self.config(), transform_config={})
        model.eval()
        cloud = self.batch()["pc_noisy"].reshape(12, 3)
        model.blend_beta = 0.0
        anchored = patch_based_denoise(
            model, cloud, patch_size=12, seed_k=1, inner_steps=2
        )
        anchor = patch_based_denoise(
            model.anchor, cloud, patch_size=12, seed_k=1, inner_steps=1
        )
        np.testing.assert_array_equal(anchored.numpy(), anchor.numpy())
        model.blend_beta = 0.75
        first, _ = model.denoise_langevin_dynamics(
            cloud.unsqueeze(0),
            point_ids=jt.arange(12).reshape(1, 12),
        )
        second, _ = model.denoise_langevin_dynamics(
            cloud.unsqueeze(0),
            point_ids=jt.arange(12).reshape(1, 12),
        )
        np.testing.assert_allclose(first.numpy(), second.numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
