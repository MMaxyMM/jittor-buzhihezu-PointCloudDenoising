import unittest

import jittor as jt
import numpy as np

from src.model.patch_inference import patch_based_denoise


class IdentityDenoiser:
    def denoise_langevin_dynamics(self, patches, num_steps=None):
        return patches, None


class FailingDenoiser:
    def denoise_langevin_dynamics(self, patches, num_steps=None):
        raise RuntimeError("expected test failure")


class PatchInferenceTest(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 0

    def test_identity_fusion_preserves_cloud(self):
        points = np.random.RandomState(123).randn(32, 3).astype(np.float32)
        output = patch_based_denoise(
            IdentityDenoiser(),
            jt.array(points),
            patch_size=16,
            seed_k=2,
            seed_k_alpha=1,
            inner_steps=1,
        )
        self.assertEqual(tuple(output.shape), points.shape)
        np.testing.assert_allclose(output.numpy(), points, atol=1e-5)

    def test_small_cloud_and_failure_are_safe(self):
        points = np.random.RandomState(123).randn(8, 3).astype(np.float32)
        output = patch_based_denoise(
            IdentityDenoiser(),
            jt.array(points),
            patch_size=16,
            seed_k=1,
        )
        self.assertEqual(tuple(output.shape), points.shape)
        failed = patch_based_denoise(
            FailingDenoiser(),
            jt.array(points),
            patch_size=8,
            seed_k=1,
        )
        self.assertIsNone(failed)


if __name__ == "__main__":
    unittest.main()
