import unittest

from select_best_checkpoint import (
    CheckpointResult,
    _mesh_path_for_cached_asset,
    rank_results,
)


class OfficialCheckpointSelectionTest(unittest.TestCase):
    def test_official_scores_rank_descending(self):
        ranked = rank_results([
            CheckpointResult("a", 1, 70.0, "ok", direction="higher"),
            CheckpointResult("b", 2, 72.0, "ok", direction="higher"),
        ])
        self.assertEqual(ranked[0].checkpoint, "b")

    def test_mesh_path_is_derived_from_cached_shapenet_path(self):
        path = _mesh_path_for_cached_asset(
            "dataset_train_pcd/shapenet/03001627/id/clean.npy",
            "dataset_train",
        )
        self.assertEqual(
            path.as_posix(),
            "dataset_train/shapenet/03001627/id/models/model_normalized.obj",
        )


if __name__ == "__main__":
    unittest.main()
