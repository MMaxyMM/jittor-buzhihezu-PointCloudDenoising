import unittest

import jittor as jt
import numpy as np

from src.model.diffusion.schedule import (
    CosineSchedule,
    q_sample,
    v_target,
    v_to_clean_and_noise,
)


class DiffusionScheduleTest(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 0

    def test_schedule_endpoints_and_monotonicity(self):
        schedule = CosineSchedule(num_train_steps=100)
        self.assertAlmostEqual(float(schedule.alphas_np[0]), 1.0, places=6)
        self.assertAlmostEqual(float(schedule.sigmas_np[0]), 0.0, places=6)
        self.assertTrue(np.all(np.diff(schedule.alphas_np) <= 1e-7))
        self.assertTrue(np.all(np.diff(schedule.sigmas_np) >= -1e-7))
        indices = schedule.inference_indices(4)
        self.assertEqual(int(indices[0]), 100)
        self.assertEqual(int(indices[-1]), 0)

    def test_v_parameterization_round_trip(self):
        schedule = CosineSchedule(num_train_steps=20)
        alpha, sigma = schedule.coefficients(np.array([7], dtype=np.int64))
        clean = jt.array(np.random.randn(1, 8, 3).astype(np.float32))
        noise = jt.array(np.random.randn(1, 8, 3).astype(np.float32))
        state = q_sample(clean, noise, alpha, sigma)
        velocity = v_target(clean, noise, alpha, sigma)
        recovered_clean, recovered_noise = v_to_clean_and_noise(
            state, velocity, alpha, sigma
        )
        np.testing.assert_allclose(
            recovered_clean.numpy(), clean.numpy(), atol=1e-5
        )
        np.testing.assert_allclose(
            recovered_noise.numpy(), noise.numpy(), atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
