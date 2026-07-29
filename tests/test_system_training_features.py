import os
import tempfile
import unittest

import jittor as jt

from src.model.residual_diffusion import ResidualDiffusionModule
from src.system.spec import DummySystem


def tiny_direct_model():
    return ResidualDiffusionModule({
        "backbone": "dgcnn",
        "frame_knn": 2,
        "feat_embedding_dim": 16,
        "decoder_hidden_dim": 8,
        "condition_embedding_dim": 8,
        "objective": "direct",
        "num_train_steps": 10,
        "patch_size": 4,
    }, {})


class SystemTrainingFeaturesTest(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 0

    def test_ema_checkpoint_state_and_model_init(self):
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_direct_model()
            system = DummySystem(
                dataset_module=None,
                model=model,
                loss_config={"loss": 1.0},
                optimizer_config={"__target__": "adam", "lr": 1e-4},
                trainer_config={
                    "epochs": 2,
                    "ema_decay": 0.9,
                    "validate_every_n_epochs": 2,
                },
            )
            system._update_ema()
            checkpoint = os.path.join(directory, "checkpoint_0.pkl")
            system._save_checkpoint(checkpoint)
            system._save_training_state(checkpoint, epoch=0)
            self.assertTrue(os.path.isfile(checkpoint))
            self.assertTrue(os.path.isfile(f"{checkpoint}.state.pkl"))

            initialized = tiny_direct_model()
            initialized_system = DummySystem(
                dataset_module=None,
                model=initialized,
                loss_config={"loss": 1.0},
                optimizer_config=None,
                trainer_config={"model_init_checkpoint": checkpoint},
            )
            self.assertEqual(initialized_system.start_epoch, 0)

            resumed = tiny_direct_model()
            resumed_system = DummySystem(
                dataset_module=None,
                model=resumed,
                loss_config={"loss": 1.0},
                optimizer_config={"__target__": "adam", "lr": 1e-4},
                trainer_config={"resume_checkpoint": checkpoint},
            )
            self.assertEqual(resumed_system.start_epoch, 1)


if __name__ == "__main__":
    unittest.main()
