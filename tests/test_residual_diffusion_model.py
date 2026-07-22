import os
import tempfile
import unittest
from copy import deepcopy

import jittor as jt
import numpy as np

from src.model.parse import get_model
from src.model.residual_diffusion import ResidualDiffusionModule


def tiny_config(backbone="dgcnn"):
    return {
        "__target__": "ResidualDiffusionModule",
        "backbone": backbone,
        "transformer_blocks": 1,
        "frame_knn": 4,
        "num_train_points": 8,
        "feat_embedding_dim": 32,
        "decoder_hidden_dim": 16,
        "condition_embedding_dim": 16,
        "num_train_steps": 20,
        "num_inference_steps": 2,
        "patch_size": 16,
        "seed_k": 1,
        "seed_k_alpha": 1,
        "lambda_l1": 0.1,
        "lambda_cd": 0.03,
        "sampling_seed": 123,
    }


class ResidualDiffusionModelTest(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 0
        np.random.seed(123)

    def _batch(self):
        noisy = jt.array(
            (np.random.randn(1, 1, 16, 3) * 0.01).astype(np.float32)
        )
        clean = jt.zeros_like(noisy)
        noise_std = jt.array(
            np.array([[[0.01]]], dtype=np.float32)
        )
        return {
            "pc_noisy": noisy,
            "pc_clean": clean,
            "noise_std": noise_std,
        }

    def test_registry_train_save_load_and_deterministic_sample(self):
        model = get_model(deepcopy(tiny_config()), transform_config={})
        self.assertIsInstance(model, ResidualDiffusionModule)
        batch = self._batch()
        optimizer = jt.optim.Adam(model.parameters(), lr=1e-4)
        loss = model.training_step(batch)["loss"]
        self.assertTrue(np.isfinite(float(loss.item())))
        optimizer.step(loss)

        model.eval()
        patches = batch["pc_noisy"].reshape(1, 16, 3)
        first, _ = model.denoise_langevin_dynamics(patches, num_steps=2)
        second, _ = model.denoise_langevin_dynamics(patches, num_steps=2)
        np.testing.assert_allclose(first.numpy(), second.numpy(), atol=1e-6)
        self.assertTrue(np.isfinite(first.numpy()).all())
        prediction = model.predict_step(
            {"pc_noisy": batch["pc_noisy"].reshape(1, 16, 3)}
        )[0]["pc_denoised"]
        self.assertEqual(prediction.shape, (16, 3))
        self.assertTrue(np.isfinite(prediction).all())

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.pkl")
            model.save(checkpoint)
            restored = get_model(
                deepcopy(tiny_config()), transform_config={}
            )
            restored.load(checkpoint)
            self.assertTrue(os.path.isfile(checkpoint))

    def test_point_transformer_forward(self):
        model = get_model(
            deepcopy(tiny_config("point_transformer")),
            transform_config={},
        )
        loss = model.training_step(self._batch())["loss"]
        self.assertTrue(np.isfinite(float(loss.item())))


if __name__ == "__main__":
    unittest.main()
