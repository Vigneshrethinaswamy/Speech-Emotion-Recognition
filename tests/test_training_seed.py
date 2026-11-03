"""Reproducibility tests for explicit training seeds."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src import train
from src.train import make_mixup_dataset


class TrainingSeedTests(unittest.TestCase):
    def test_train_model_propagates_seed_to_rngs_and_mixup(self):
        x = np.zeros((2, 3, 1), dtype=np.float32)
        y = np.eye(8, dtype=np.float32)[:2]
        data = (x, x, x, y, y, y, {i: 1.0 for i in range(8)})
        history = MagicMock()
        history.history = {"accuracy": [0.5], "loss": [1.0]}
        model = MagicMock()
        model.fit.return_value = history

        with patch.object(train, "load_and_prepare_data", return_value=data), \
             patch.object(train, "build_model", return_value=model), \
             patch.object(train, "build_callbacks", return_value=[]), \
             patch.object(train, "plot_training_history"), \
             patch.object(train, "make_mixup_dataset", return_value="dataset") as mixup, \
             patch.object(train.tf.keras.utils, "set_random_seed") as tf_seed, \
             patch.object(train.random, "seed") as python_seed, \
             patch.object(train.np.random, "seed") as numpy_seed:
            train.train_model(seed=19)

        tf_seed.assert_called_once_with(19)
        python_seed.assert_called_once_with(19)
        numpy_seed.assert_called_once_with(19)
        self.assertEqual(mixup.call_args.kwargs["seed"], 19)

    def test_mixup_dataset_is_repeatable_for_a_seed(self):
        x = np.arange(24, dtype=np.float32).reshape(8, 3, 1)
        y = np.eye(2, dtype=np.float32)[np.arange(8) % 2]
        weights = np.ones(8, dtype=np.float32)

        def collect(seed):
            return [tuple(value.numpy() for value in batch)
                    for batch in make_mixup_dataset(x, y, weights, 4, seed=seed)]

        first = collect(7)
        repeated = collect(7)
        different = collect(8)
        for first_batch, repeated_batch in zip(first, repeated):
            for left, right in zip(first_batch, repeated_batch):
                np.testing.assert_array_equal(left, right)
        self.assertTrue(any(
            not np.array_equal(left, right)
            for first_batch, other_batch in zip(first, different)
            for left, right in zip(first_batch, other_batch)
        ))


if __name__ == "__main__":
    unittest.main()

