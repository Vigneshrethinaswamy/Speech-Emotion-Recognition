"""
main.py
========

End-to-end pipeline entry point for the CodeAlpha_EmotionRecognitionFromSpeech
project.

Pipeline stages
----------------
    1. Feature Extraction  - Discover RAVDESS files, split them by source
                              file (GroupShuffleSplit, BEFORE augmentation
                              exists) into train/validation/test, then extract
                              features -- with augmentation + SpecAugment
                              for training files only, and clean
                              extraction for validation and test files.
                              Skipped if processed features already exist,
                              unless --force-extract is passed.
    2. Data Loading         - Load and prepare cached splits once during
                              training; fit StandardScaler on training only.
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

from src.evaluate import evaluate_model
from src.feature_extraction import (
    DEFAULT_X_TRAIN_PATH,
    PROJECT_ROOT,
    cache_is_valid,
    dataset_fingerprint,
    process_dataset,
    save_features,
)
from src.train import train_model

FULL_RUN_DIR = PROJECT_ROOT / "outputs" / "full_run"
FULL_RUN_MODEL_PATH = FULL_RUN_DIR / "models" / "best_model.keras"
FULL_RUN_TRAINING_DIR = FULL_RUN_DIR / "training"
FULL_RUN_EVALUATION_DIR = FULL_RUN_DIR / "evaluation"

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
        default=str(PROJECT_ROOT / "data" / "ravdess"),
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
        default=32,
        help="Training batch size (default: 32).",
    )
    return parser


def run_pipeline(
    data_dir: str | Path = PROJECT_ROOT / "data" / "ravdess",
    force_extract: bool = False,
    epochs: int = 50,
    batch_size: int = 32,
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
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    fingerprint = dataset_fingerprint(data_dir)
    features_exist = cache_is_valid(fingerprint)

    if features_exist and not force_extract:
        logger.info(
            "Validated actor-disjoint cache found at '%s'; "
            "skipping extraction. Use --force-extract to recompute.",
            DEFAULT_X_TRAIN_PATH.parent,
        )
    else:
        logger.info("=== STAGE 1: Feature Extraction (group-split, leakage-safe) ===")
        X_train, X_val, X_test, y_train, y_val, y_test = process_dataset(data_dir=data_dir)
        save_features(X_train, X_val, X_test, y_train, y_val, y_test, fingerprint=fingerprint)

    # ----------------------------------------------------------------- #
    # Data loading, scaling, and encoding happen once inside train_model;
    # it returns the prepared held-out arrays for final evaluation.
    logger.info("=== STAGE 2 & 3: Data Preparation, Model Building + Training ===")
    model, history, X_test, y_test = train_model(
        epochs=epochs,
        batch_size=batch_size,
        model_path=FULL_RUN_MODEL_PATH,
        plots_dir=FULL_RUN_TRAINING_DIR,
        return_test_data=True,
    )

    # ----------------------------------------------------------------- #
    # Stage 5: Evaluation
    # ----------------------------------------------------------------- #
    logger.info("=== STAGE 5: Evaluation ===")
    test_accuracy, report, confusion = evaluate_model(
        model, X_test, y_test, plots_dir=FULL_RUN_EVALUATION_DIR
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Final Test Accuracy: %.4f (%.2f%%)", test_accuracy, test_accuracy * 100)
    logger.info("Model saved to: %s", FULL_RUN_MODEL_PATH)
    logger.info("Training plots saved to: %s", FULL_RUN_TRAINING_DIR)
    logger.info("Evaluation reports saved to: %s", FULL_RUN_EVALUATION_DIR)
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

