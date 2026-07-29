import unittest

import numpy as np

from src.data.asset import Asset
from src.data.augment import AugmentPatch, AugmentSample


class CleanNormalsTest(unittest.TestCase):
    def test_cached_sampling_patch_and_rotation_keep_normals_aligned(self):
        points = np.arange(90, dtype=np.float32).reshape(30, 3) / 90.0
        normals = np.tile(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (30, 1)
        )
        asset = Asset(
            sampled_vertices=points.copy(),
            sampled_normals=normals.copy(),
        )
        np.random.seed(1)
        AugmentSample(num_samples=20).apply(asset)
        asset.sampled_vertices_noisy = asset.sampled_vertices.copy()
        asset.meta = {"noise_std": np.float32(0.01)}
        AugmentPatch(
            patch_size=8,
            num_patches=1,
            train_cvm_network=False,
            center_mode="noisy_seed",
        ).apply(asset)
        self.assertEqual(asset.meta["clean_normals"].shape, (1, 8, 3))
        np.testing.assert_allclose(
            np.linalg.norm(asset.meta["clean_normals"], axis=-1), 1.0
        )

    def test_normal_transform_uses_inverse_transpose_and_renormalizes(self):
        asset = Asset(
            sampled_vertices=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            sampled_normals=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
        )
        transform = np.eye(4, dtype=np.float32)
        transform[0, 0] = 2.0
        asset.transform(transform)
        expected = np.array([[0.5, 1.0, 0.0]], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(asset.sampled_normals, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
