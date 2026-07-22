import unittest

import numpy as np

from src.data.asset import Asset
from src.data.augment import AugmentAddNoise, AugmentPatch


class DiffusionDataTest(unittest.TestCase):
    def test_noise_scale_reaches_patch_metadata(self):
        np.random.seed(123)
        points = np.random.randn(64, 3).astype(np.float32)
        asset = Asset(sampled_vertices=points)
        AugmentAddNoise(
            noise_std_min=0.01,
            noise_std_max=0.01,
            noise_type="laplace",
        ).apply(asset)
        self.assertAlmostEqual(float(asset.meta["noise_std"]), 0.01, places=6)

        AugmentPatch(
            patch_size=16,
            num_patches=2,
            train_cvm_network=False,
        ).apply(asset)
        self.assertEqual(asset.meta["noise_std"].shape, (2, 1))
        np.testing.assert_allclose(asset.meta["noise_std"], 0.01, atol=1e-6)
        self.assertEqual(asset.meta["pc_noisy"].shape, (2, 16, 3))
        self.assertEqual(asset.meta["pc_clean"].shape, (2, 16, 3))

    def test_cvm_metadata_remains_available(self):
        np.random.seed(123)
        points = np.random.randn(64, 3).astype(np.float32)
        asset = Asset(sampled_vertices=points)
        AugmentAddNoise(0.01, 0.01, "laplace").apply(asset)
        AugmentPatch(16, 2, True).apply(asset)
        for key in (
            "pc_noisy",
            "pc_clean",
            "seed_points_t",
            "original_time_step",
            "noise_std",
        ):
            self.assertIn(key, asset.meta)


if __name__ == "__main__":
    unittest.main()
