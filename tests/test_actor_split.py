"""Regression tests for speaker-disjoint RAVDESS splitting."""

import unittest
from pathlib import Path

from src.feature_extraction import split_dataset_files


class ActorDisjointSplitTests(unittest.TestCase):
    def setUp(self):
        self.paths = [
            Path(f"data/ravdess/Actor_{actor:02d}/03-01-01-01-01-01-{clip:02d}.wav")
            for actor in range(1, 9)
            for clip in range(1, 4)
        ]
        self.labels = [clip % 8 for actor in range(1, 9) for clip in range(1, 4)]

    def test_no_actor_appears_in_both_splits(self):
        train_paths, test_paths, _, _ = split_dataset_files(
            self.paths, self.labels, test_size=0.25, random_state=42
        )
        train_actors = {path.parent.name for path in train_paths}
        test_actors = {path.parent.name for path in test_paths}

        self.assertTrue(train_actors)
        self.assertTrue(test_actors)
        self.assertTrue(train_actors.isdisjoint(test_actors))
        self.assertEqual(set(train_paths) | set(test_paths), set(self.paths))

    def test_rejects_paths_without_actor_directory(self):
        paths = list(self.paths)
        paths[0] = Path("data/ravdess/misc/recording.wav")

        with self.assertRaisesRegex(ValueError, "Actor_XX"):
            split_dataset_files(paths, self.labels, test_size=0.25, random_state=42)


if __name__ == "__main__":
    unittest.main()
