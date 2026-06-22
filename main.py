"""
main.py
========

End-to-end pipeline entry point for the CodeAlpha_EmotionRecognitionFromSpeech
project.

Pipeline stages
----------------
    1. Feature Extraction  - Discover RAVDESS files, split them by source
                              file (GroupShuffleSplit, BEFORE augmentation
                              exists) into train/test, then extract
                              features -- with augmentation + SpecAugment
                              for the training files only, and clean
                              (unaugmented) extraction for the test files.
                              Skipped if processed features already exist,
                              unless --force-extract is passed.
    2. Data Loading         - Load the pre-split arrays, standardize
                              (StandardScaler fit on train only), reshape
                              for the CNN. No splitting happens here.
    3. Model Building       - Construct and compile the Conv1D-BiLSTM CNN.
    4. Training              - Train with Mixup (training batches only),
                              EarlyStopping, ReduceLROnPlateau, and
                              ModelCheckpoint callbacks; save plots.
    5. Evaluation            - Compute test accuracy, classification report,
                              and confusion matrix on the held-out, clean
                              (unaugmented) test split.

Usage
-----
    python main.py
    python main.py --force-extract
    python main.py --data-dir data/ravdess --epochs 30

Author: Senior ML Engineering Team
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tensorflow as tf

from src.data_loader import load_and_prepare_data
from src.evaluate import evaluate_model
from src.feature_extraction import (
    DEFAULT_X_TEST_PATH,
    DEFAULT_X_TRAIN_PATH,
    DEFAULT_Y_TEST_PATH,
    DEFAULT_Y_TRAIN_PATH,
    process_dataset,
    save_features,
)
from src.train import DEFAULT_MODEL_PATH, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full Emotion Recognition From Speech pipeline."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ravdess",
        help="Path to the root RAVDESS dataset directory (default: data/ravdess).",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of MFCC features even if cached .npy files exist.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 32).",
    )
    return parser


def run_pipeline(
    data_dir: str = "data/ravdess",
    force_extract: bool = False,
    epochs: int = 50,
    batch_size: int = 64,
) -> float:
    """
    Execute the full Feature Extraction -> Data Loading -> Model Building ->
    Training -> Evaluation pipeline.

    Parameters
    ----------
    data_dir : str, default="data/ravdess"
        Root directory of the raw RAVDESS dataset.
    force_extract : bool, default=False
        If True, re-run feature extraction even if cached features exist.
    epochs : int, default=50
        Maximum number of training epochs.
    batch_size : int, default=32
        Training batch size.

    Returns
    -------
    float
        The final test accuracy achieved by the trained model.
    """
    # ----------------------------------------------------------------- #
    # Stage 1: Feature Extraction (leakage-safe: split happens on the
    # original files BEFORE augmentation; see feature_extraction.py)
    # ----------------------------------------------------------------- #
    features_exist = (
        DEFAULT_X_TRAIN_PATH.exists()
        and DEFAULT_X_TEST_PATH.exists()
        and DEFAULT_Y_TRAIN_PATH.exists()
        and DEFAULT_Y_TEST_PATH.exists()
    )

    if features_exist and not force_extract:
        logger.info(
            "Cached pre-split features found at '%s', '%s', '%s', '%s'; "
            "skipping extraction. Use --force-extract to recompute.",
            DEFAULT_X_TRAIN_PATH,
            DEFAULT_X_TEST_PATH,
            DEFAULT_Y_TRAIN_PATH,
            DEFAULT_Y_TEST_PATH,
        )
    else:
        logger.info("=== STAGE 1: Feature Extraction (group-split, leakage-safe) ===")
        X_train, X_test, y_train, y_test = process_dataset(data_dir=data_dir)
        save_features(X_train, X_test, y_train, y_test)

    # ----------------------------------------------------------------- #
    # Stage 2: Data Loading
    # ----------------------------------------------------------------- #
    logger.info("=== STAGE 2: Data Loading ===")
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # ----------------------------------------------------------------- #
    # Stage 3 + 4: Model Building + Training
    # ----------------------------------------------------------------- #
    logger.info("=== STAGE 3 & 4: Model Building + Training ===")
    model, history = train_model(epochs=epochs, batch_size=batch_size)

    # ----------------------------------------------------------------- #
    # Stage 5: Evaluation
    # ----------------------------------------------------------------- #
    logger.info("=== STAGE 5: Evaluation ===")
    test_accuracy, report, confusion = evaluate_model(model, X_test, y_test)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Final Test Accuracy: %.4f (%.2f%%)", test_accuracy, test_accuracy * 100)
    logger.info("Model saved to: %s", DEFAULT_MODEL_PATH)
    logger.info("Plots saved to: plots/")
    logger.info("=" * 60)

    return test_accuracy


if __name__ == "__main__":
    arg_parser = _build_arg_parser()
    args = arg_parser.parse_args()

    final_accuracy = run_pipeline(
        data_dir=args.data_dir,
        force_extract=args.force_extract,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 60)
    print(f"FINAL TEST ACCURACY: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print("=" * 60)