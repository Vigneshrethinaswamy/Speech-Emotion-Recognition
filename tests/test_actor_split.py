"""Regression tests for speaker-disjoint RAVDESS splitting."""

import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pathlib import Path

from src import feature_extraction as features
from src.feature_extraction import split_dataset_files
from src.data_loader import fit_and_save_scaler, reshape_for_cnn


class ActorDisjointSplitTests(unittest.TestCase):
    def setUp(self):
        self.paths = [
            Path(f"data/ravdess/Actor_{actor:02d}/03-01-01-01-01-01-{clip:02d}.wav")
            for actor in range(1, 9)
            for clip in range(1, 4)
        ]
        self.labels = [clip % 8 for actor in range(1, 9) for clip in range(1, 4)]

    def test_no_actor_appears_in_both_splits(self):
        train_paths, val_paths, test_paths, *_ = split_dataset_files(
            self.paths, self.labels, test_size=0.25, random_state=42
        )
        train_actors = {path.parent.name for path in train_paths}
        val_actors = {path.parent.name for path in val_paths}
        test_actors = {path.parent.name for path in test_paths}

        self.assertTrue(train_actors)
        self.assertTrue(test_actors)
        self.assertTrue(train_actors.isdisjoint(val_actors))
        self.assertTrue(train_actors.isdisjoint(test_actors))
        self.assertTrue(val_actors.isdisjoint(test_actors))
        self.assertEqual(set(train_paths) | set(val_paths) | set(test_paths), set(self.paths))

    def test_rejects_paths_without_actor_directory(self):
        paths = list(self.paths)
        paths[0] = Path("data/ravdess/misc/recording.wav")

        with self.assertRaisesRegex(ValueError, "Actor_XX"):
            split_dataset_files(paths, self.labels, test_size=0.25, random_state=42)

    def test_cache_requires_matching_metadata_and_valid_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ["X_train.npy", "X_val.npy", "X_test.npy", "y_train.npy", "y_val.npy", "y_test.npy"]
            arrays = [np.zeros((3, features.FEATURE_DIM)), np.zeros((2, features.FEATURE_DIM)),
                      np.zeros((2, features.FEATURE_DIM)), np.zeros(3, dtype=np.int64),
                      np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)]
            paths = [root / name for name in names]
            for path, array in zip(paths, arrays):
                np.save(path, array)
            metadata_path = root / "cache_metadata.json"
            metadata_path.write_text(json.dumps({"schema_version": 3,
                "split": "actor-disjoint-70/15/15-v3", "fingerprint": "abc",
                "feature_dim": features.FEATURE_DIM,
                "shapes": [list(a.shape) for a in arrays],
                "array_sha256": [features._sha256_file(path) for path in paths]}), encoding="utf-8")
            with patch.object(features, "DEFAULT_CACHE_METADATA_PATH", metadata_path), \
                 patch.object(features, "DEFAULT_X_TRAIN_PATH", paths[0]), \
                 patch.object(features, "DEFAULT_X_VAL_PATH", paths[1]), \
                 patch.object(features, "DEFAULT_X_TEST_PATH", paths[2]), \
                 patch.object(features, "DEFAULT_Y_TRAIN_PATH", paths[3]), \
                 patch.object(features, "DEFAULT_Y_VAL_PATH", paths[4]), \
                 patch.object(features, "DEFAULT_Y_TEST_PATH", paths[5]):
                loaded = features.load_features(paths[0], paths[2], paths[3], paths[5])
                self.assertEqual([a.shape for a in loaded], [a.shape for a in arrays])
                self.assertTrue(features.cache_is_valid("abc"))
                self.assertFalse(features.cache_is_valid("different-source"))
                arrays[1] = np.zeros((2, features.FEATURE_DIM + 1))
                np.save(paths[1], arrays[1])
                self.assertFalse(features.cache_is_valid("abc"))

    def test_feature_reshape_has_cnn_input_shape(self):
        shaped = reshape_for_cnn(np.zeros((5, features.FEATURE_DIM), dtype=np.float64))
        self.assertEqual(shaped.shape, (5, features.FEATURE_DIM, 1))
        self.assertEqual(shaped.dtype, np.float32)

    def test_scaler_is_fit_from_training_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            train = np.array([[1.0, 10.0], [3.0, 14.0]])
            scaler = fit_and_save_scaler(train, Path(temp) / "scaler.pkl")
            np.testing.assert_allclose(scaler.mean_, [2.0, 12.0])


if __name__ == "__main__":
    unittest.main()

